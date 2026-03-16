from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
DEFAULT_VEHICLE_ASSET = DEFAULT_VENDOR_DIR / "kenney_raceCarRed.glb"
DEFAULT_TEXTURE_DIR = DEFAULT_VENDOR_DIR / "polyhaven"
ASPHALT_DIFFUSE = DEFAULT_TEXTURE_DIR / "asphalt_pit_lane_diff_1k.jpg"
ASPHALT_ROUGHNESS = DEFAULT_TEXTURE_DIR / "asphalt_pit_lane_rough_1k.jpg"
ASPHALT_NORMAL = DEFAULT_TEXTURE_DIR / "asphalt_pit_lane_nor_gl_1k.png"
GRASS_DIFFUSE = DEFAULT_TEXTURE_DIR / "sparse_grass_diff_1k.jpg"
GRASS_ROUGHNESS = DEFAULT_TEXTURE_DIR / "sparse_grass_rough_1k.jpg"
GRASS_NORMAL = DEFAULT_TEXTURE_DIR / "sparse_grass_nor_gl_1k.png"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render a replay bundle in Blender.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution-x", type=int, default=1920)
    parser.add_argument("--resolution-y", type=int, default=1080)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--save-blend-path", default=None)
    parser.add_argument("--vehicle-asset", default=None)
    return parser.parse_args(argv)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("World")


def read_centerline(path: Path) -> list[Vector]:
    points: list[Vector] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(Vector((float(row["x"]), float(row["y"]), 0.0)))
    return points


def compute_normals(points: list[Vector], closed: bool) -> list[Vector]:
    normals: list[Vector] = []
    count = len(points)
    for index in range(count):
        prev_index = (index - 1) % count if closed else max(index - 1, 0)
        next_index = (index + 1) % count if closed else min(index + 1, count - 1)
        tangent = points[next_index] - points[prev_index]
        if tangent.length < 1e-9:
            tangent = Vector((1.0, 0.0, 0.0))
        tangent.normalize()
        normals.append(Vector((-tangent.y, tangent.x, 0.0)))
    return normals


def create_strip_mesh(
    name: str,
    inner: list[Vector],
    outer: list[Vector],
    z_inner: float,
    z_outer: float,
    material: bpy.types.Material,
    closed: bool,
) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    count = len(inner)
    for index in range(count):
        verts.append((inner[index].x, inner[index].y, z_inner))
        verts.append((outer[index].x, outer[index].y, z_outer))
    segment_count = count if closed else count - 1
    for index in range(segment_count):
        next_index = (index + 1) % count
        base = 2 * index
        next_base = 2 * next_index
        faces.append((base, base + 1, next_base + 1, next_base))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name=f"{name}UV")
    cumulative = [0.0]
    for index in range(1, count):
        cumulative.append(cumulative[-1] + (inner[index] - inner[index - 1]).length)
    if closed and count > 1:
        total_length = cumulative[-1] + (inner[0] - inner[-1]).length
    else:
        total_length = max(cumulative[-1], 1.0)
    loops_per_face = 4
    for face_index, face in enumerate(faces):
        index = face_index
        next_index = (index + 1) % count
        u0 = cumulative[index] / 3.0
        u1 = (cumulative[next_index] if next_index > index else total_length) / 3.0
        loop_base = face_index * loops_per_face
        face_uvs = ((u0, 0.0), (u0, 1.0), (u1, 1.0), (u1, 0.0))
        for offset, uv in enumerate(face_uvs):
            uv_layer.data[loop_base + offset].uv = uv

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
    *,
    clearcoat: float = 0.0,
    alpha: float | None = None,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = clearcoat
    elif "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = clearcoat
    if alpha is not None and "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = alpha
        material.blend_method = "BLEND"
        if hasattr(material, "shadow_method"):
            material.shadow_method = "HASHED"
    return material


def make_paint_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    bump = nodes.new(type="ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (7.0, 7.0, 7.0)
    noise.inputs["Scale"].default_value = 42.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.42
    color_ramp.color_ramp.elements[0].position = 0.34
    color_ramp.color_ramp.elements[0].color = (
        max(color[0] * 0.58, 0.0),
        max(color[1] * 0.58, 0.0),
        max(color[2] * 0.58, 0.0),
        1.0,
    )
    color_ramp.color_ramp.elements[1].position = 0.86
    color_ramp.color_ramp.elements[1].color = color
    bump.inputs["Strength"].default_value = 0.018

    vector_output = tex_coord.outputs["UV"] if "UV" in tex_coord.outputs else tex_coord.outputs["Object"]
    links.new(vector_output, mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    principled.inputs["Roughness"].default_value = 0.16
    principled.inputs["Metallic"].default_value = 0.42
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = 1.0
        principled.inputs["Coat Roughness"].default_value = 0.08
    elif "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = 1.0
        if "Clearcoat Roughness" in principled.inputs:
            principled.inputs["Clearcoat Roughness"].default_value = 0.08
    return material


def make_glass_material(name: str = "GlassMatHero") -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "BLEND"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "HASHED"
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.04, 0.06, 0.08, 1.0)
    principled.inputs["Roughness"].default_value = 0.03
    if "Transmission Weight" in principled.inputs:
        principled.inputs["Transmission Weight"].default_value = 0.72
    elif "Transmission" in principled.inputs:
        principled.inputs["Transmission"].default_value = 0.72
    if "IOR" in principled.inputs:
        principled.inputs["IOR"].default_value = 1.45
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = 0.38
    return material


def make_tire_material(name: str = "TireMatHero") -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    wave = nodes.new(type="ShaderNodeTexWave")
    mix = nodes.new(type="ShaderNodeMixRGB")
    bump = nodes.new(type="ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (30.0, 30.0, 30.0)
    noise.inputs["Scale"].default_value = 14.0
    noise.inputs["Detail"].default_value = 6.0
    wave.inputs["Scale"].default_value = 48.0
    wave.inputs["Distortion"].default_value = 4.0
    mix.inputs["Fac"].default_value = 0.28
    bump.inputs["Strength"].default_value = 0.08

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(noise.outputs["Color"], mix.inputs["Color1"])
    links.new(wave.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    principled.inputs["Base Color"].default_value = (0.02, 0.02, 0.023, 1.0)
    principled.inputs["Roughness"].default_value = 0.86
    return material


def make_metal_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    return make_material(name, color, roughness=0.22, metallic=0.88, clearcoat=0.12)


def load_image(path: Path) -> bpy.types.Image | None:
    if not path.exists():
        return None
    existing = bpy.data.images.get(path.name)
    if existing is not None:
        return existing
    return bpy.data.images.load(str(path))


def make_pbr_material(
    name: str,
    *,
    base_color_path: Path,
    roughness_path: Path | None = None,
    normal_path: Path | None = None,
    scale: float = 4.0,
    metallic: float = 0.0,
    specular: float = 0.5,
) -> bpy.types.Material | None:
    base_image = load_image(base_color_path)
    if base_image is None:
        return None

    roughness_image = load_image(roughness_path) if roughness_path is not None else None
    normal_image = load_image(normal_path) if normal_path is not None else None

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    color_tex = nodes.new(type="ShaderNodeTexImage")
    color_tex.image = base_image
    color_tex.interpolation = "Linear"
    if hasattr(color_tex, "projection"):
        color_tex.projection = "BOX"
        color_tex.projection_blend = 0.16
    links.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])
    links.new(color_tex.outputs["Color"], principled.inputs["Base Color"])

    if roughness_image is not None:
        roughness_tex = nodes.new(type="ShaderNodeTexImage")
        roughness_tex.image = roughness_image
        roughness_tex.image.colorspace_settings.name = "Non-Color"
        roughness_tex.interpolation = "Linear"
        if hasattr(roughness_tex, "projection"):
            roughness_tex.projection = "BOX"
            roughness_tex.projection_blend = 0.16
        links.new(mapping.outputs["Vector"], roughness_tex.inputs["Vector"])
        links.new(roughness_tex.outputs["Color"], principled.inputs["Roughness"])
    else:
        principled.inputs["Roughness"].default_value = 0.72

    if normal_image is not None:
        normal_tex = nodes.new(type="ShaderNodeTexImage")
        normal_tex.image = normal_image
        normal_tex.image.colorspace_settings.name = "Non-Color"
        normal_tex.interpolation = "Linear"
        if hasattr(normal_tex, "projection"):
            normal_tex.projection = "BOX"
            normal_tex.projection_blend = 0.16
        normal_map = nodes.new(type="ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.6
        links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
        links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

    principled.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = specular
    elif "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = specular
    return material


def make_grass_material(name: str = "GrassMat") -> bpy.types.Material:
    textured = make_pbr_material(
        name,
        base_color_path=GRASS_DIFFUSE,
        roughness_path=GRASS_ROUGHNESS,
        normal_path=GRASS_NORMAL,
        scale=1.0,
        metallic=0.0,
        specular=0.24,
    )
    if textured is not None:
        return textured
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise_fine = nodes.new(type="ShaderNodeTexNoise")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    mix = nodes.new(type="ShaderNodeMixRGB")
    bump = nodes.new(type="ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (28.0, 28.0, 28.0)
    noise.inputs["Scale"].default_value = 5.4
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.58
    noise_fine.inputs["Scale"].default_value = 46.0
    noise_fine.inputs["Detail"].default_value = 12.0
    noise_fine.inputs["Roughness"].default_value = 0.48
    color_ramp.color_ramp.elements[0].position = 0.32
    color_ramp.color_ramp.elements[0].color = (0.05, 0.09, 0.035, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.82
    color_ramp.color_ramp.elements[1].color = (0.24, 0.33, 0.14, 1.0)
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.42
    bump.inputs["Strength"].default_value = 0.09

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_fine.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], mix.inputs["Color1"])
    links.new(noise_fine.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], principled.inputs["Base Color"])
    links.new(noise_fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    principled.inputs["Roughness"].default_value = 0.98
    return material


def make_asphalt_material(name: str = "RoadMat") -> bpy.types.Material:
    textured = make_pbr_material(
        name,
        base_color_path=ASPHALT_DIFFUSE,
        roughness_path=ASPHALT_ROUGHNESS,
        normal_path=ASPHALT_NORMAL,
        scale=1.0,
        metallic=0.0,
        specular=0.34,
    )
    if textured is not None:
        return textured
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise_fine = nodes.new(type="ShaderNodeTexNoise")
    noise_stain = nodes.new(type="ShaderNodeTexNoise")
    wave = nodes.new(type="ShaderNodeTexWave")
    mix = nodes.new(type="ShaderNodeMixRGB")
    mix_stain = nodes.new(type="ShaderNodeMixRGB")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    bump = nodes.new(type="ShaderNodeBump")

    mapping.inputs["Scale"].default_value = (18.0, 18.0, 18.0)
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 11.0
    noise.inputs["Roughness"].default_value = 0.6
    noise_fine.inputs["Scale"].default_value = 55.0
    noise_fine.inputs["Detail"].default_value = 14.0
    noise_fine.inputs["Roughness"].default_value = 0.45
    noise_stain.inputs["Scale"].default_value = 3.4
    noise_stain.inputs["Detail"].default_value = 10.0
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 8.0
    wave.inputs["Distortion"].default_value = 14.0
    color_ramp.color_ramp.elements[0].position = 0.24
    color_ramp.color_ramp.elements[0].color = (0.045, 0.045, 0.05, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.90
    color_ramp.color_ramp.elements[1].color = (0.17, 0.17, 0.18, 1.0)
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.4
    mix_stain.blend_type = "MIX"
    mix_stain.inputs["Fac"].default_value = 0.18
    mix_stain.inputs["Color2"].default_value = (0.06, 0.06, 0.06, 1.0)
    bump.inputs["Strength"].default_value = 0.045

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_fine.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_stain.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], mix.inputs["Color1"])
    links.new(noise_fine.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], mix_stain.inputs["Color1"])
    links.new(noise_stain.outputs["Fac"], mix_stain.inputs["Fac"])
    links.new(mix_stain.outputs["Color"], principled.inputs["Base Color"])
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    principled.inputs["Roughness"].default_value = 0.88
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = 0.12
    elif "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = 0.12
    return material


def apply_smooth_and_bevel(obj: bpy.types.Object, bevel_width: float = 0.03) -> None:
    if obj.type == "MESH":
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        modifier = obj.modifiers.new(name="Bevel", type="BEVEL")
        modifier.width = bevel_width
        modifier.segments = 2


def create_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel_width: float = 0.03,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    apply_smooth_and_bevel(obj, bevel_width=bevel_width)
    return obj


def create_cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    parent: bpy.types.Object | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 24,
    bevel_width: float = 0.01,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    apply_smooth_and_bevel(obj, bevel_width=bevel_width)
    return obj


def create_tree(name: str, location: tuple[float, float, float]) -> None:
    trunk_mat = make_material(f"{name}TrunkMat", (0.16, 0.10, 0.06, 1.0), roughness=0.92)
    foliage_mat = make_material(f"{name}FoliageMat", (0.09, 0.17, 0.09, 1.0), roughness=0.82)
    trunk = create_cylinder(
        f"{name}Trunk",
        (location[0], location[1], location[2] + 1.1),
        radius=0.18,
        depth=2.2,
        material=trunk_mat,
        rotation=(0.0, 0.0, 0.0),
        vertices=10,
        bevel_width=0.0,
    )
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.45, location=(location[0], location[1], location[2] + 3.0))
    foliage = bpy.context.active_object
    foliage.name = f"{name}Foliage"
    foliage.scale = (1.35, 1.35, 1.7)
    foliage.data.materials.append(foliage_mat)
    apply_smooth_and_bevel(foliage, bevel_width=0.0)
    foliage.parent = trunk


def create_hill(name: str, location: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    hill_mat = make_material(f"{name}Mat", (0.20, 0.24, 0.18, 1.0), roughness=1.0)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=location)
    hill = bpy.context.active_object
    hill.name = name
    hill.scale = scale
    hill.data.materials.append(hill_mat)
    apply_smooth_and_bevel(hill, bevel_width=0.0)


def create_grandstand(name: str, location: Vector, heading: float) -> None:
    support_mat = make_metal_material(f"{name}SupportMat", (0.38, 0.40, 0.44, 1.0))
    seat_mat = make_material(f"{name}SeatMat", (0.10, 0.14, 0.20, 1.0), roughness=0.56)
    banner_mat = make_material(f"{name}BannerMat", (0.86, 0.88, 0.92, 1.0), roughness=0.42, clearcoat=0.18)
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)

    for index in range(6):
        depth = 8.8 - index * 1.05
        height = 0.32 + index * 0.18
        create_box(
            f"{name}Step_{index}",
            (-0.4 - index * 0.55, 0.0, 0.35 + height * 0.5),
            (1.45, depth, height),
            seat_mat,
            parent=root,
            bevel_width=0.02,
        )
    for side in (-1.0, 1.0):
        for index in range(4):
            create_cylinder(
                f"{name}Column_{side}_{index}",
                (-1.8 + index * 1.2, side * 8.6, 2.3),
                radius=0.10,
                depth=4.6,
                material=support_mat,
                parent=root,
                vertices=10,
                bevel_width=0.0,
            )
    create_box(
        f"{name}Roof",
        (-0.9, 0.0, 4.92),
        (3.8, 9.8, 0.16),
        banner_mat,
        parent=root,
        bevel_width=0.02,
    )


def create_bridge(name: str, location: Vector, heading: float) -> None:
    support_mat = make_metal_material(f"{name}SupportMat", (0.55, 0.57, 0.61, 1.0))
    sign_mat = make_material(f"{name}SignMat", (0.95, 0.96, 0.98, 1.0), roughness=0.24, clearcoat=0.26)
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    for side in (-1.0, 1.0):
        create_box(
            f"{name}Column_{side}",
            (0.0, side * 6.3, 2.8),
            (0.22, 0.22, 2.8),
            support_mat,
            parent=root,
            bevel_width=0.02,
        )
    create_box(
        f"{name}Beam",
        (0.0, 0.0, 5.46),
        (0.34, 7.2, 0.24),
        support_mat,
        parent=root,
        bevel_width=0.02,
    )
    create_box(
        f"{name}Sign",
        (0.0, 0.0, 4.55),
        (0.18, 5.8, 0.72),
        sign_mat,
        parent=root,
        bevel_width=0.02,
    )


def create_light_pole(name: str, location: Vector, heading: float) -> None:
    pole_mat = make_metal_material(f"{name}PoleMat", (0.54, 0.56, 0.60, 1.0))
    lamp_mat = make_material(f"{name}LampMat", (0.94, 0.95, 0.90, 1.0), roughness=0.18, clearcoat=0.3)
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    create_cylinder(
        f"{name}Pole",
        (0.0, 0.0, 5.0),
        radius=0.08,
        depth=10.0,
        material=pole_mat,
        parent=root,
        vertices=12,
        bevel_width=0.0,
    )
    create_box(
        f"{name}Arm",
        (0.0, 0.75, 9.4),
        (0.06, 0.78, 0.05),
        pole_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        f"{name}Lamp",
        (0.0, 1.42, 9.18),
        (0.12, 0.22, 0.08),
        lamp_mat,
        parent=root,
        bevel_width=0.01,
    )


def create_tent(name: str, location: Vector, heading: float) -> None:
    pole_mat = make_metal_material(f"{name}PoleMat", (0.60, 0.62, 0.66, 1.0))
    canopy_mat = make_material(f"{name}CanopyMat", (0.92, 0.93, 0.95, 1.0), roughness=0.28, clearcoat=0.18)
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    for x_coord in (-1.2, 1.2):
        for y_coord in (-1.2, 1.2):
            create_cylinder(
                f"{name}Pole_{x_coord}_{y_coord}",
                (x_coord, y_coord, 1.4),
                radius=0.05,
                depth=2.8,
                material=pole_mat,
                parent=root,
                vertices=10,
                bevel_width=0.0,
            )
    create_box(
        f"{name}Roof",
        (0.0, 0.0, 2.9),
        (1.55, 1.55, 0.10),
        canopy_mat,
        parent=root,
        bevel_width=0.02,
    )


def create_service_vehicle(name: str, location: Vector, heading: float, color: tuple[float, float, float, float]) -> None:
    body_mat = make_paint_material(f"{name}BodyMat", color)
    trim_mat = make_metal_material(f"{name}TrimMat", (0.22, 0.24, 0.28, 1.0))
    glass_mat = make_glass_material(f"{name}GlassMat")
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    create_box(f"{name}Body", (0.0, 0.0, 0.72), (1.8, 0.82, 0.55), body_mat, parent=root, bevel_width=0.03)
    create_box(f"{name}Cab", (0.35, 0.0, 1.25), (0.78, 0.78, 0.42), body_mat, parent=root, bevel_width=0.03)
    create_box(f"{name}Glass", (0.72, 0.0, 1.22), (0.18, 0.72, 0.26), glass_mat, parent=root, bevel_width=0.01)
    for x_coord in (-1.05, 1.05):
        for y_coord in (-0.72, 0.72):
            create_cylinder(
                f"{name}Wheel_{x_coord}_{y_coord}",
                (x_coord, y_coord, 0.38),
                radius=0.34,
                depth=0.22,
                material=make_tire_material(f"{name}TireMat_{x_coord}_{y_coord}"),
                parent=root,
                rotation=(math.pi * 0.5, 0.0, 0.0),
                vertices=24,
                bevel_width=0.0,
            )
    create_box(f"{name}Bumper", (1.66, 0.0, 0.56), (0.10, 0.84, 0.18), trim_mat, parent=root, bevel_width=0.01)


def create_marshall_post(name: str, location: Vector, heading: float) -> None:
    wall_mat = make_material(f"{name}WallMat", (0.86, 0.87, 0.90, 1.0), roughness=0.34, clearcoat=0.12)
    roof_mat = make_material(f"{name}RoofMat", (0.16, 0.18, 0.20, 1.0), roughness=0.52)
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler = (0.0, 0.0, heading)
    create_box(f"{name}Base", (0.0, 0.0, 1.1), (1.1, 0.85, 1.0), wall_mat, parent=root, bevel_width=0.03)
    create_box(f"{name}Window", (0.0, 0.78, 1.35), (0.88, 0.08, 0.34), make_glass_material(f"{name}Glass"), parent=root, bevel_width=0.01)
    create_box(f"{name}Roof", (0.0, 0.0, 2.26), (1.32, 1.04, 0.10), roof_mat, parent=root, bevel_width=0.02)


def create_trackside_scene(track_points: list[Vector], normals: list[Vector], road_half: float) -> None:
    if not track_points:
        return

    support_mat = make_metal_material("FenceSupportMat", (0.62, 0.64, 0.68, 1.0))
    fence_mat = make_material("FenceMat", (0.62, 0.68, 0.74, 0.22), roughness=0.24, metallic=0.82, alpha=0.22)
    building_mat = make_material("PitMat", (0.72, 0.74, 0.78, 1.0), roughness=0.52, clearcoat=0.12)
    dark_building = make_material("PitTrimMat", (0.10, 0.12, 0.15, 1.0), roughness=0.58)
    tire_stack_mat = make_tire_material("BarrierTireMat")

    segment_stride = max(12, len(track_points) // 42)
    for point_index in range(0, len(track_points), segment_stride):
        point = track_points[point_index]
        normal = normals[point_index]
        tangent_index = min(point_index + 3, len(track_points) - 1)
        tangent = track_points[tangent_index] - point
        if tangent.length < 1e-6:
            tangent = Vector((1.0, 0.0, 0.0))
        tangent.normalize()
        heading = math.atan2(tangent.y, tangent.x)

        for side in (-1.0, 1.0):
            base = point + normal * (side * (road_half + 3.4))
            create_cylinder(
                f"FencePost_{point_index}_{side}",
                (base.x, base.y, 1.35),
                radius=0.05,
                depth=2.7,
                material=support_mat,
                rotation=(0.0, 0.0, 0.0),
                vertices=8,
                bevel_width=0.0,
            )
            create_box(
                f"FencePanel_{point_index}_{side}",
                (base.x, base.y, 1.55),
                (0.12, 1.65, 1.0),
                fence_mat,
                rotation=(0.0, 0.0, heading + math.pi * 0.5),
                bevel_width=0.004,
            )
            if point_index % (segment_stride * 3) == 0:
                create_light_pole(
                    f"LightPole_{point_index}_{side}",
                    base + normal * (side * 2.8),
                    heading,
                )

    anchor = track_points[0]
    next_anchor = track_points[min(10, len(track_points) - 1)]
    tangent = next_anchor - anchor
    if tangent.length < 1e-6:
        tangent = Vector((1.0, 0.0, 0.0))
    tangent.normalize()
    heading = math.atan2(tangent.y, tangent.x)
    normal = Vector((-tangent.y, tangent.x, 0.0))

    special_indices = [0, len(track_points) // 4, len(track_points) // 2, (3 * len(track_points)) // 4]
    for stand_index, point_index in enumerate(special_indices, start=1):
        point = track_points[point_index]
        normal = normals[point_index]
        tangent = track_points[min(point_index + 8, len(track_points) - 1)] - point
        if tangent.length < 1e-6:
            tangent = Vector((1.0, 0.0, 0.0))
        tangent.normalize()
        local_heading = math.atan2(tangent.y, tangent.x)
        create_grandstand(
            f"Grandstand_{stand_index}",
            point + normal * (road_half + 20.0 + 2.0 * (stand_index % 2)),
            local_heading,
        )
        create_bridge(
            f"Bridge_{stand_index}",
            point + tangent * 8.0,
            local_heading,
        )
        create_marshall_post(
            f"Marshall_{stand_index}",
            point - normal * (road_half + 7.2),
            local_heading,
        )

    pit_center = anchor + tangent * 18.0 - normal * (road_half + 14.5)
    pit_root = bpy.data.objects.new("PitBuilding", None)
    bpy.context.scene.collection.objects.link(pit_root)
    pit_root.location = pit_center
    pit_root.rotation_euler = (0.0, 0.0, heading + math.pi)
    create_box("PitLower", (0.0, 0.0, 1.0), (8.6, 3.8, 1.0), building_mat, parent=pit_root, bevel_width=0.04)
    create_box("PitUpper", (-1.2, 0.0, 3.2), (6.8, 3.2, 0.72), building_mat, parent=pit_root, bevel_width=0.03)
    create_box("PitGlass", (-0.2, 0.0, 3.1), (5.2, 2.8, 0.42), make_glass_material("PitGlassMat"), parent=pit_root, bevel_width=0.01)
    create_box("PitTrim", (6.8, 0.0, 1.5), (0.20, 4.0, 1.6), dark_building, parent=pit_root, bevel_width=0.02)
    pit_wall_mat = make_material("PitWallMat", (0.82, 0.84, 0.88, 1.0), roughness=0.42, clearcoat=0.08)
    for index in range(12):
        wall_center = anchor + tangent * (12.0 + index * 5.2) - normal * (road_half + 4.2)
        create_box(
            f"PitWall_{index}",
            (wall_center.x, wall_center.y, 0.62),
            (2.45, 0.22, 0.62),
            pit_wall_mat,
            rotation=(0.0, 0.0, heading),
            bevel_width=0.02,
        )
    for index in range(5):
        create_tent(
            f"PaddockTent_{index}",
            pit_center + Vector((-8.0 - index * 3.4, 8.8, 0.0)),
            heading + math.pi * 0.5,
        )
    for index, color in enumerate(
        (
            (0.88, 0.18, 0.16, 1.0),
            (0.94, 0.94, 0.95, 1.0),
            (0.18, 0.26, 0.84, 1.0),
        ),
        start=1,
    ):
        create_service_vehicle(
            f"ServiceVehicle_{index}",
            pit_center + Vector((-5.0 - index * 4.6, -7.5 + index * 2.2, 0.0)),
            heading + math.pi,
            color,
        )
    for index in range(5):
        board_center = anchor + tangent * (18.0 + index * 11.0) + normal * (road_half + 7.0)
        create_box(
            f"TrackBanner_{index}",
            (board_center.x, board_center.y, 1.8),
            (1.8, 0.10, 0.76),
            make_material(f"TrackBannerMat_{index}", (0.96, 0.97, 0.99, 1.0), roughness=0.28, clearcoat=0.18),
            rotation=(0.0, 0.0, heading),
            bevel_width=0.02,
        )

    for stack_index in range(8):
        base = anchor + tangent * (30.0 + stack_index * 2.2) - normal * (road_half + 2.1)
        for layer in range(3):
            create_cylinder(
                f"TireBarrier_{stack_index}_{layer}",
                (base.x, base.y, 0.34 + layer * 0.26),
                radius=0.34,
                depth=0.26,
                material=tire_stack_mat,
                rotation=(math.pi * 0.5, 0.0, heading),
                vertices=18,
                bevel_width=0.0,
            )

    radius = max(max(abs(point.x) for point in track_points), max(abs(point.y) for point in track_points)) + 120.0
    hill_specs = (
        ((radius * 0.68, radius * 0.24, 12.0), (120.0, 60.0, 26.0)),
        ((-radius * 0.42, radius * 0.74, 16.0), (160.0, 70.0, 34.0)),
        ((radius * 0.20, -radius * 0.72, 14.0), (140.0, 58.0, 28.0)),
        ((-radius * 0.76, -radius * 0.26, 18.0), (180.0, 80.0, 38.0)),
    )
    for index, (location, scale) in enumerate(hill_specs, start=1):
        create_hill(f"Hill_{index}", location, scale)


def create_empty(name: str, parent: bpy.types.Object | None, location: Vector | tuple[float, float, float]) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def import_online_vehicle(vehicle_spec: dict, vehicle_asset: Path | None = None) -> dict[str, bpy.types.Object] | None:
    asset_path = Path(vehicle_asset) if vehicle_asset is not None else DEFAULT_VEHICLE_ASSET
    if not asset_path.exists():
        return None

    existing_names = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(asset_path))
    imported_objects = [obj for obj in bpy.data.objects if obj.name not in existing_names]
    if not imported_objects:
        return None

    model_root = next((obj for obj in imported_objects if obj.type == "EMPTY" and obj.parent is None), None)
    if model_root is None:
        return None

    wheel_names = {
        "front_left": "wheelFrontLeft",
        "front_right": "wheelFrontRight",
        "rear_left": "wheelBackLeft",
        "rear_right": "wheelBackRight",
    }
    wheel_meshes: dict[str, bpy.types.Object] = {}
    for key, needle in wheel_names.items():
        match = next((obj for obj in imported_objects if obj.type == "MESH" and needle.lower() in obj.name.lower()), None)
        if match is None:
            return None
        wheel_meshes[key] = match

    source_radius = max(wheel_meshes["front_left"].dimensions.y, wheel_meshes["front_left"].dimensions.z) * 0.5
    desired_radius = float(vehicle_spec["wheel_radius"])
    scale = desired_radius / max(source_radius, 1e-6)
    wheel_center_z = float(wheel_meshes["front_left"].location.z)
    ground_offset = desired_radius - wheel_center_z * scale

    sim_root = bpy.data.objects.new("VehicleRoot", None)
    bpy.context.scene.collection.objects.link(sim_root)
    model_root.parent = sim_root
    model_root.location = (0.0, 0.0, ground_offset)
    model_root.rotation_mode = "XYZ"
    # The Kenney car points along local -Y, while the simulator uses +X as forward.
    model_root.rotation_euler = (0.0, 0.0, math.pi * 0.5)
    model_root.scale = (scale, scale, scale)

    front_left_steer = create_empty("FrontLeftSteer", model_root, wheel_meshes["front_left"].location.copy())
    front_right_steer = create_empty("FrontRightSteer", model_root, wheel_meshes["front_right"].location.copy())
    front_left_spin = create_empty("FrontLeftSpin", front_left_steer, (0.0, 0.0, 0.0))
    front_right_spin = create_empty("FrontRightSpin", front_right_steer, (0.0, 0.0, 0.0))
    rear_left_spin = create_empty("RearLeftSpin", model_root, wheel_meshes["rear_left"].location.copy())
    rear_right_spin = create_empty("RearRightSpin", model_root, wheel_meshes["rear_right"].location.copy())

    for key, spin_parent in {
        "front_left": front_left_spin,
        "front_right": front_right_spin,
        "rear_left": rear_left_spin,
        "rear_right": rear_right_spin,
    }.items():
        wheel = wheel_meshes[key]
        wheel.parent = spin_parent
        wheel.location = (0.0, 0.0, 0.0)
        wheel.rotation_euler = (0.0, 0.0, 0.0)

    tune_imported_vehicle_materials(imported_objects)

    for obj in imported_objects:
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

    return {
        "root": sim_root,
        "model_root": model_root,
        "steer_left": front_left_steer,
        "steer_right": front_right_steer,
        "front_left_spin": front_left_spin,
        "front_right_spin": front_right_spin,
        "rear_left_spin": rear_left_spin,
        "rear_right_spin": rear_right_spin,
        "front_left": wheel_meshes["front_left"],
        "front_right": wheel_meshes["front_right"],
        "rear_left": wheel_meshes["rear_left"],
        "rear_right": wheel_meshes["rear_right"],
        "wheel_spin_axis": "X",
        "wheel_spin_sign": 1.0,
    }


def tune_imported_vehicle_materials(imported_objects: list[bpy.types.Object]) -> None:
    replacements: dict[str, bpy.types.Material] = {}
    for obj in imported_objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            if material.name in replacements:
                slot.material = replacements[material.name]
                continue
            material_name = material.name.lower()
            if "cartire" in material_name:
                replacement = make_tire_material(f"{material.name}_Hero")
            elif material_name == "grey":
                replacement = make_metal_material(f"{material.name}_Hero", (0.58, 0.61, 0.66, 1.0))
            elif "glass" in material_name:
                replacement = make_glass_material(f"{material.name}_Hero")
            else:
                replacement = make_paint_material(f"{material.name}_Hero", (0.80, 0.34, 0.36, 1.0))
            slot.material = replacement
            replacements[material.name] = replacement


def add_ground(size: float = 900.0) -> None:
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, -0.03))
    plane = bpy.context.active_object
    plane.name = "Ground"
    plane.data.materials.append(make_grass_material())


def build_track(track_points: list[Vector], track_width: float, closed: bool) -> None:
    normals = compute_normals(track_points, closed)
    road_half = 0.5 * track_width

    road_inner = [point - normal * road_half for point, normal in zip(track_points, normals)]
    road_outer = [point + normal * road_half for point, normal in zip(track_points, normals)]

    shoulder_inner = [point - normal * (road_half + 1.2) for point, normal in zip(track_points, normals)]
    shoulder_outer = [point + normal * (road_half + 1.2) for point, normal in zip(track_points, normals)]

    curb_left_inner = [point + normal * (road_half + 0.08) for point, normal in zip(track_points, normals)]
    curb_left_outer = [point + normal * (road_half + 0.48) for point, normal in zip(track_points, normals)]
    curb_right_inner = [point - normal * (road_half + 0.08) for point, normal in zip(track_points, normals)]
    curb_right_outer = [point - normal * (road_half + 0.48) for point, normal in zip(track_points, normals)]

    shoulder_mat = make_material("ShoulderMat", (0.16, 0.16, 0.17, 1.0), roughness=0.96)
    road_mat = make_asphalt_material()
    curb_red = make_material("CurbRed", (0.78, 0.10, 0.08, 1.0), roughness=0.54, clearcoat=0.24)
    curb_white = make_material("CurbWhite", (0.96, 0.96, 0.95, 1.0), roughness=0.44, clearcoat=0.20)
    stripe_mat = make_material("StripeMat", (0.96, 0.96, 0.96, 1.0), roughness=0.28, clearcoat=0.12)
    guardrail_mat = make_metal_material("GuardrailMat", (0.76, 0.79, 0.82, 1.0))
    runoff_mat = make_material("RunoffMat", (0.22, 0.34, 0.20, 1.0), roughness=0.92)

    create_strip_mesh("Shoulder", shoulder_inner, shoulder_outer, 0.0, 0.0, shoulder_mat, closed)
    create_strip_mesh("Road", road_inner, road_outer, 0.015, 0.015, road_mat, closed)
    create_strip_mesh("CurbLeft", curb_left_inner, curb_left_outer, 0.018, 0.018, curb_red, closed)
    create_strip_mesh("CurbRight", curb_right_outer, curb_right_inner, 0.018, 0.018, curb_white, closed)

    left_stripe_inner = [point + normal * (road_half - 0.10) for point, normal in zip(track_points, normals)]
    left_stripe_outer = [point + normal * (road_half - 0.02) for point, normal in zip(track_points, normals)]
    right_stripe_inner = [point - normal * (road_half - 0.02) for point, normal in zip(track_points, normals)]
    right_stripe_outer = [point - normal * (road_half - 0.10) for point, normal in zip(track_points, normals)]
    create_strip_mesh("StripeLeft", left_stripe_inner, left_stripe_outer, 0.019, 0.019, stripe_mat, closed)
    create_strip_mesh("StripeRight", right_stripe_outer, right_stripe_inner, 0.019, 0.019, stripe_mat, closed)

    left_guard_inner = [point + normal * (road_half + 1.1) for point, normal in zip(track_points, normals)]
    left_guard_outer = [point + normal * (road_half + 1.25) for point, normal in zip(track_points, normals)]
    right_guard_inner = [point - normal * (road_half + 1.1) for point, normal in zip(track_points, normals)]
    right_guard_outer = [point - normal * (road_half + 1.25) for point, normal in zip(track_points, normals)]
    create_strip_mesh("GuardrailLeft", left_guard_inner, left_guard_outer, 0.32, 0.38, guardrail_mat, closed)
    create_strip_mesh("GuardrailRight", right_guard_outer, right_guard_inner, 0.38, 0.32, guardrail_mat, closed)
    runoff_left_inner = [point + normal * (road_half + 0.48) for point, normal in zip(track_points, normals)]
    runoff_left_outer = [point + normal * (road_half + 1.02) for point, normal in zip(track_points, normals)]
    runoff_right_inner = [point - normal * (road_half + 0.48) for point, normal in zip(track_points, normals)]
    runoff_right_outer = [point - normal * (road_half + 1.02) for point, normal in zip(track_points, normals)]
    create_strip_mesh("RunoffLeft", runoff_left_inner, runoff_left_outer, 0.012, 0.012, runoff_mat, closed)
    create_strip_mesh("RunoffRight", runoff_right_outer, runoff_right_inner, 0.012, 0.012, runoff_mat, closed)

    anchor = track_points[0]
    next_anchor = track_points[min(12, len(track_points) - 1)]
    tangent = next_anchor - anchor
    if tangent.length < 1e-9:
        tangent = Vector((1.0, 0.0, 0.0))
    tangent.normalize()
    normal = Vector((-tangent.y, tangent.x, 0.0))
    sponsor_mat = make_material("SponsorMat", (0.94, 0.94, 0.96, 1.0), roughness=0.24, clearcoat=0.22)
    sponsor_trim = make_material("SponsorTrimMat", (0.08, 0.22, 0.68, 1.0), roughness=0.16, metallic=0.18, clearcoat=0.36)
    for index, shift in enumerate((36.0, 78.0, 120.0, 162.0), start=1):
        base = anchor + tangent * shift + normal * (road_half + 14.0)
        board = create_box(
            f"SponsorBoard_{index}",
            (base.x, base.y, 1.4),
            (2.6, 0.12, 1.1),
            sponsor_mat,
            rotation=(0.0, 0.0, math.atan2(tangent.y, tangent.x)),
            bevel_width=0.02,
        )
        create_box(
            f"SponsorBoardTrim_{index}",
            (0.0, 0.0, 0.0),
            (2.75, 0.06, 0.14),
            sponsor_trim,
            parent=board,
            bevel_width=0.01,
        )

    tree_indices = range(8, len(track_points), max(12, len(track_points) // 28))
    for index, point_index in enumerate(tree_indices, start=1):
        point = track_points[point_index]
        normal = normals[point_index]
        offset = road_half + 34.0 + (index % 3) * 4.0
        create_tree(f"TreeLeft_{index}", (point.x + normal.x * offset, point.y + normal.y * offset, 0.0))
        create_tree(f"TreeRight_{index}", (point.x - normal.x * (offset + 10.0), point.y - normal.y * (offset + 10.0), 0.0))
        if index % 2 == 0:
            create_tree(
                f"TreeClusterLeft_{index}",
                (point.x + normal.x * (offset + 6.5), point.y + normal.y * (offset + 6.5), 0.0),
            )
        if index % 3 == 0:
            create_tree(
                f"TreeClusterRight_{index}",
                (point.x - normal.x * (offset + 15.0), point.y - normal.y * (offset + 15.0), 0.0),
            )

    create_trackside_scene(track_points, normals, road_half)


def look_at_rotation(camera_location: Vector, target_location: Vector):
    direction = target_location - camera_location
    if direction.length < 1e-9:
        direction = Vector((1.0, 0.0, -0.2))
    return direction.to_track_quat("-Z", "Y").to_euler()


def create_procedural_vehicle(vehicle_spec: dict) -> dict[str, bpy.types.Object]:
    chassis_x, chassis_y, chassis_z = vehicle_spec["chassis_size"]
    wheel_radius = float(vehicle_spec["wheel_radius"])
    lf = float(vehicle_spec["lf"])
    lr = float(vehicle_spec["lr"])
    track_half = max(0.5, chassis_y * 0.52)

    root = bpy.data.objects.new("VehicleRoot", None)
    bpy.context.scene.collection.objects.link(root)

    body_mat = make_material("BodyMat", (0.05, 0.24, 0.82, 1.0), roughness=0.24, metallic=0.18)
    accent_mat = make_material("AccentMat", (0.86, 0.10, 0.10, 1.0), roughness=0.3, metallic=0.05)
    carbon_mat = make_material("CarbonMat", (0.03, 0.03, 0.035, 1.0), roughness=0.65, metallic=0.12)
    glass_mat = make_material("GlassMat", (0.10, 0.14, 0.18, 1.0), roughness=0.08, metallic=0.0)
    wheel_mat = make_material("WheelMat", (0.03, 0.03, 0.03, 1.0), roughness=0.88)

    body = create_box(
        "VehicleBody",
        (-0.15, 0.0, wheel_radius + chassis_z * 0.50),
        (chassis_x * 0.26, chassis_y * 0.46, chassis_z * 0.26),
        body_mat,
        parent=root,
        bevel_width=0.05,
    )
    create_box(
        "SidePodLeft",
        (-0.35, track_half * 0.47, wheel_radius + chassis_z * 0.42),
        (0.92, 0.28, 0.20),
        body_mat,
        parent=root,
        bevel_width=0.04,
    )
    create_box(
        "SidePodRight",
        (-0.35, -track_half * 0.47, wheel_radius + chassis_z * 0.42),
        (0.92, 0.28, 0.20),
        body_mat,
        parent=root,
        bevel_width=0.04,
    )
    create_box(
        "EngineCover",
        (-0.78, 0.0, wheel_radius + chassis_z * 0.72),
        (0.68, 0.38, 0.28),
        body_mat,
        parent=root,
        bevel_width=0.04,
    )
    create_box(
        "Nose",
        (1.02, 0.0, wheel_radius + chassis_z * 0.36),
        (1.20, 0.14, 0.10),
        body_mat,
        parent=root,
        bevel_width=0.03,
    )
    create_box(
        "FrontWing",
        (1.74, 0.0, wheel_radius + 0.12),
        (0.58, chassis_y * 0.72, 0.035),
        carbon_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        "RearWingMain",
        (-1.74, 0.0, wheel_radius + chassis_z * 1.28),
        (0.26, chassis_y * 0.76, 0.035),
        carbon_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        "RearWingFlap",
        (-1.62, 0.0, wheel_radius + chassis_z * 1.12),
        (0.20, chassis_y * 0.72, 0.03),
        accent_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        "HaloBar",
        (0.12, 0.0, wheel_radius + chassis_z * 1.10),
        (0.26, 0.03, 0.03),
        carbon_mat,
        parent=root,
        bevel_width=0.01,
    )
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.34, location=(0.18, 0.0, wheel_radius + chassis_z * 0.95))
    canopy = bpy.context.active_object
    canopy.name = "VehicleCanopy"
    canopy.scale = (1.22, 0.86, 0.74)
    canopy.parent = root
    canopy.data.materials.append(glass_mat)
    apply_smooth_and_bevel(canopy, bevel_width=0.0)
    create_box(
        "AccentStripe",
        (-0.10, 0.0, wheel_radius + chassis_z * 0.70),
        (1.18, 0.06, 0.03),
        accent_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        "RearCrashStructure",
        (-2.06, 0.0, wheel_radius + 0.42),
        (0.20, 0.10, 0.12),
        carbon_mat,
        parent=root,
        bevel_width=0.01,
    )
    create_box(
        "FrontWingLeftPlate",
        (1.74, track_half * 0.88, wheel_radius + 0.18),
        (0.06, 0.18, 0.12),
        carbon_mat,
        parent=root,
        bevel_width=0.008,
    )
    create_box(
        "FrontWingRightPlate",
        (1.74, -track_half * 0.88, wheel_radius + 0.18),
        (0.06, 0.18, 0.12),
        carbon_mat,
        parent=root,
        bevel_width=0.008,
    )
    create_box(
        "MirrorLeft",
        (0.46, track_half * 0.42, wheel_radius + chassis_z * 0.86),
        (0.12, 0.05, 0.04),
        accent_mat,
        parent=root,
        rotation=(0.0, 0.0, math.radians(20.0)),
        bevel_width=0.01,
    )
    create_box(
        "MirrorRight",
        (0.46, -track_half * 0.42, wheel_radius + chassis_z * 0.86),
        (0.12, 0.05, 0.04),
        accent_mat,
        parent=root,
        rotation=(0.0, 0.0, math.radians(-20.0)),
        bevel_width=0.01,
    )
    create_cylinder(
        "RollHoop",
        (-0.52, 0.0, wheel_radius + chassis_z * 1.28),
        radius=0.05,
        depth=0.32,
        material=carbon_mat,
        parent=root,
        rotation=(math.pi * 0.5, 0.0, 0.0),
        vertices=16,
        bevel_width=0.0,
    )

    steer_left = bpy.data.objects.new("FrontLeftSteer", None)
    steer_right = bpy.data.objects.new("FrontRightSteer", None)
    bpy.context.scene.collection.objects.link(steer_left)
    bpy.context.scene.collection.objects.link(steer_right)
    steer_left.parent = root
    steer_right.parent = root
    steer_left.location = (lf, track_half, wheel_radius)
    steer_right.location = (lf, -track_half, wheel_radius)

    def add_wheel(name: str, parent: bpy.types.Object, location: tuple[float, float, float]) -> tuple[bpy.types.Object, bpy.types.Object]:
        spin = bpy.data.objects.new(f"{name}Spin", None)
        bpy.context.scene.collection.objects.link(spin)
        spin.parent = parent
        spin.location = location
        bpy.ops.mesh.primitive_cylinder_add(vertices=36, radius=wheel_radius, depth=0.28, location=(0.0, 0.0, 0.0), rotation=(math.pi * 0.5, 0.0, 0.0))
        wheel = bpy.context.active_object
        wheel.name = name
        wheel.parent = spin
        wheel.location = (0.0, 0.0, 0.0)
        wheel.data.materials.append(wheel_mat)
        apply_smooth_and_bevel(wheel, bevel_width=0.01)
        rim = create_cylinder(
            f"{name}Rim",
            (0.0, 0.0, 0.0),
            radius=wheel_radius * 0.52,
            depth=0.30,
            material=make_material(f"{name}RimMat", (0.55, 0.58, 0.62, 1.0), roughness=0.32, metallic=0.72),
            parent=spin,
            rotation=(math.pi * 0.5, 0.0, 0.0),
            vertices=24,
            bevel_width=0.004,
        )
        rim.scale = (1.0, 1.0, 0.12)
        return spin, wheel

    create_cylinder(
        "FrontLeftPushrod",
        (lf * 0.55, track_half * 0.65, wheel_radius + 0.22),
        radius=0.03,
        depth=1.10,
        material=carbon_mat,
        parent=root,
        rotation=(0.0, math.radians(58.0), math.radians(18.0)),
        vertices=12,
        bevel_width=0.0,
    )
    create_cylinder(
        "FrontRightPushrod",
        (lf * 0.55, -track_half * 0.65, wheel_radius + 0.22),
        radius=0.03,
        depth=1.10,
        material=carbon_mat,
        parent=root,
        rotation=(0.0, math.radians(58.0), math.radians(-18.0)),
        vertices=12,
        bevel_width=0.0,
    )

    front_left_spin, front_left = add_wheel("FrontLeftWheel", steer_left, (0.0, 0.0, 0.0))
    front_right_spin, front_right = add_wheel("FrontRightWheel", steer_right, (0.0, 0.0, 0.0))
    rear_left_spin, rear_left = add_wheel("RearLeftWheel", root, (-lr, track_half, wheel_radius))
    rear_right_spin, rear_right = add_wheel("RearRightWheel", root, (-lr, -track_half, wheel_radius))

    return {
        "root": root,
        "steer_left": steer_left,
        "steer_right": steer_right,
        "front_left_spin": front_left_spin,
        "front_right_spin": front_right_spin,
        "rear_left_spin": rear_left_spin,
        "rear_right_spin": rear_right_spin,
        "front_left": front_left,
        "front_right": front_right,
        "rear_left": rear_left,
        "rear_right": rear_right,
        "wheel_spin_axis": "Y",
        "wheel_spin_sign": -1.0,
    }


def create_vehicle(vehicle_spec: dict, vehicle_asset: Path | None = None) -> dict[str, bpy.types.Object]:
    imported = import_online_vehicle(vehicle_spec, vehicle_asset=vehicle_asset)
    if imported is not None:
        return imported
    return create_procedural_vehicle(vehicle_spec)


def animate_vehicle(vehicle: dict[str, bpy.types.Object], trajectory: list[dict]) -> None:
    root = vehicle["root"]
    steer_left = vehicle["steer_left"]
    steer_right = vehicle["steer_right"]
    wheel_spins = [
        vehicle["front_left_spin"],
        vehicle["front_right_spin"],
        vehicle["rear_left_spin"],
        vehicle["rear_right_spin"],
    ]
    wheel_spin_axis = str(vehicle.get("wheel_spin_axis", "Y")).upper()
    wheel_spin_sign = float(vehicle.get("wheel_spin_sign", -1.0))

    for frame_number, row in enumerate(trajectory, start=1):
        root.location = (float(row["x"]), float(row["y"]), 0.0)
        root.rotation_euler = (0.0, 0.0, float(row["yaw"]))
        root.keyframe_insert(data_path="location", frame=frame_number)
        root.keyframe_insert(data_path="rotation_euler", frame=frame_number)

        steering_angle = float(row.get("steering_angle", 0.0))
        steer_left.rotation_euler = (0.0, 0.0, steering_angle)
        steer_right.rotation_euler = (0.0, 0.0, steering_angle)
        steer_left.keyframe_insert(data_path="rotation_euler", frame=frame_number)
        steer_right.keyframe_insert(data_path="rotation_euler", frame=frame_number)

        wheel_rotation = float(row.get("wheel_rotation", 0.0))
        for spin in wheel_spins:
            spin_value = wheel_spin_sign * wheel_rotation
            if wheel_spin_axis == "X":
                spin.rotation_euler = (spin_value, 0.0, 0.0)
            elif wheel_spin_axis == "Z":
                spin.rotation_euler = (0.0, 0.0, spin_value)
            else:
                spin.rotation_euler = (0.0, spin_value, 0.0)
            spin.keyframe_insert(data_path="rotation_euler", frame=frame_number)


def make_line_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    emission_strength: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "BLEND"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "NONE"
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = color[3]
    if "Emission Color" in principled.inputs:
        principled.inputs["Emission Color"].default_value = color
    if "Emission Strength" in principled.inputs:
        principled.inputs["Emission Strength"].default_value = emission_strength
    principled.inputs["Roughness"].default_value = 0.28
    return material


def create_curve_overlay(
    name: str,
    trajectories_xy: list[list[tuple[float, float]]],
    *,
    z_offset: float,
    bevel_depth: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object | None:
    valid_trajectories = [points for points in trajectories_xy if len(points) >= 2]
    if not valid_trajectories:
        return None

    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.fill_mode = "FULL"
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2

    for points in valid_trajectories:
        spline = curve.splines.new(type="POLY")
        spline.points.add(len(points) - 1)
        for index, (x_coord, y_coord) in enumerate(points):
            spline.points[index].co = (float(x_coord), float(y_coord), z_offset, 1.0)

    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def keyframe_overlay_visibility(obj: bpy.types.Object, frame_number: int) -> None:
    visible_frame = max(1, int(frame_number))
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=max(0, visible_frame - 1))
    obj.keyframe_insert(data_path="hide_viewport", frame=max(0, visible_frame - 1))

    obj.hide_render = False
    obj.hide_viewport = False
    obj.keyframe_insert(data_path="hide_render", frame=visible_frame)
    obj.keyframe_insert(data_path="hide_viewport", frame=visible_frame)

    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=visible_frame + 1)
    obj.keyframe_insert(data_path="hide_viewport", frame=visible_frame + 1)


def planner_xy_lists(raw_points) -> list[list[tuple[float, float]]]:
    output: list[list[tuple[float, float]]] = []
    for candidate in raw_points:
        points = []
        for point in candidate:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if len(points) >= 2:
            output.append(points)
    return output


def create_planner_debug_overlays(trajectory: list[dict]) -> None:
    overlay_collection = bpy.data.collections.new("PlannerDebug")
    bpy.context.scene.collection.children.link(overlay_collection)
    candidate_material = make_line_material(
        "PlannerCandidateMat",
        (0.18, 0.56, 1.0, 0.16),
        emission_strength=1.9,
    )
    final_material = make_line_material(
        "PlannerFinalMat",
        (1.0, 0.48, 0.06, 0.98),
        emission_strength=4.0,
    )

    for frame_number, row in enumerate(trajectory, start=1):
        planner_debug = row.get("planner_debug")
        if not planner_debug:
            continue

        candidate_xy = planner_xy_lists(planner_debug.get("candidate_xy", [])[:100])
        final_xy = planner_xy_lists([planner_debug.get("final_xy", [])])

        candidate_obj = create_curve_overlay(
            f"PlannerCandidates_{frame_number:04d}",
            candidate_xy,
            z_offset=0.055,
            bevel_depth=0.015,
            material=candidate_material,
            collection=overlay_collection,
        )
        final_obj = create_curve_overlay(
            f"PlannerFinal_{frame_number:04d}",
            final_xy,
            z_offset=0.075,
            bevel_depth=0.032,
            material=final_material,
            collection=overlay_collection,
        )
        if candidate_obj is not None:
            keyframe_overlay_visibility(candidate_obj, frame_number)
        if final_obj is not None:
            keyframe_overlay_visibility(final_obj, frame_number)


def add_lights() -> None:
    bpy.ops.object.light_add(type="SUN", location=(48.0, -62.0, 92.0))
    sun = bpy.context.active_object
    sun.data.energy = 4.9
    sun.data.angle = math.radians(0.75)
    sun.rotation_euler = (math.radians(40.0), math.radians(2.0), math.radians(18.0))

    bpy.ops.object.light_add(type="AREA", location=(-18.0, 14.0, 8.0))
    fill = bpy.context.active_object
    fill.data.energy = 1800.0
    fill.data.shape = "RECTANGLE"
    fill.data.size = 16.0
    fill.data.size_y = 8.0
    fill.rotation_euler = (math.radians(70.0), 0.0, math.radians(-108.0))

    bpy.ops.object.light_add(type="AREA", location=(12.0, -10.0, 6.0))
    rim = bpy.context.active_object
    rim.data.energy = 850.0
    rim.data.shape = "RECTANGLE"
    rim.data.size = 10.0
    rim.data.size_y = 6.0
    rim.rotation_euler = (math.radians(88.0), 0.0, math.radians(64.0))

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    background = nodes.new(type="ShaderNodeBackground")
    sky = nodes.new(type="ShaderNodeTexSky")
    world_output = nodes.new(type="ShaderNodeOutputWorld")
    sky_types = {item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items}
    if "NISHITA" in sky_types:
        sky.sky_type = "NISHITA"
    elif "HOSEK_WILKIE" in sky_types:
        sky.sky_type = "HOSEK_WILKIE"
    else:
        sky.sky_type = "PREETHAM"
    if "Sun Elevation" in sky.inputs:
        sky.inputs["Sun Elevation"].default_value = math.radians(27.0)
    if "Sun Rotation" in sky.inputs:
        sky.inputs["Sun Rotation"].default_value = math.radians(18.0)
    if "Air" in sky.inputs:
        sky.inputs["Air"].default_value = 1.1
    if "Dust" in sky.inputs:
        sky.inputs["Dust"].default_value = 0.4
    if "Ozone" in sky.inputs:
        sky.inputs["Ozone"].default_value = 0.4
    background.inputs["Strength"].default_value = 0.8
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], world_output.inputs["Surface"])


def configure_compositor(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    if not hasattr(scene, "node_tree"):
        return
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    render_layers = nodes.new(type="CompositorNodeRLayers")
    glare = nodes.new(type="CompositorNodeGlare")
    rgb_curves = nodes.new(type="CompositorNodeCurveRGB")
    lens = nodes.new(type="CompositorNodeLensdist")
    vignette_mix = nodes.new(type="CompositorNodeMixRGB")
    ellipse = nodes.new(type="CompositorNodeEllipseMask")
    blur = nodes.new(type="CompositorNodeBlur")
    composite = nodes.new(type="CompositorNodeComposite")

    glare.glare_type = "FOG_GLOW"
    glare.threshold = 0.88
    glare.size = 5
    lens.inputs["Dispersion"].default_value = 0.008
    if "Distort" in lens.inputs:
        lens.inputs["Distort"].default_value = 0.015
    vignette_mix.blend_type = "MULTIPLY"
    vignette_mix.inputs["Fac"].default_value = 0.18
    ellipse.width = 0.88
    ellipse.height = 0.82
    blur.filter_type = "GAUSS"
    blur.size_x = 260
    blur.size_y = 260

    # Slight contrast lift.
    curve = rgb_curves.mapping.curves[3]
    curve.points[0].location = (0.0, 0.0)
    curve.points[1].location = (0.42, 0.38)
    curve.points[2].location = (0.78, 0.86)
    curve.points[3].location = (1.0, 1.0)

    links.new(render_layers.outputs["Image"], glare.inputs["Image"])
    links.new(glare.outputs["Image"], rgb_curves.inputs["Image"])
    links.new(rgb_curves.outputs["Image"], lens.inputs["Image"])
    links.new(lens.outputs["Image"], vignette_mix.inputs["Color1"])
    links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
    links.new(blur.outputs["Image"], vignette_mix.inputs["Color2"])
    links.new(vignette_mix.outputs["Image"], composite.inputs["Image"])


def configure_render(scene: bpy.types.Scene, args: argparse.Namespace, fps: int, frame_end: int) -> None:
    requested_engine = str(args.engine)
    available_engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if requested_engine not in available_engines and requested_engine == "BLENDER_EEVEE_NEXT":
        requested_engine = "BLENDER_EEVEE"
    scene.render.engine = requested_engine
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.fps = fps
    scene.render.resolution_x = int(args.resolution_x)
    scene.render.resolution_y = int(args.resolution_y)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_motion_blur = False
    available_views = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    if "AgX - Medium High Contrast" in available_views:
        scene.view_settings.look = "AgX - Medium High Contrast"
    elif "High Contrast" in available_views:
        scene.view_settings.look = "High Contrast"
    else:
        scene.view_settings.look = "None"
    if hasattr(scene.view_settings, "exposure"):
        scene.view_settings.exposure = 0.15

    if requested_engine == "CYCLES":
        scene.cycles.samples = int(args.samples)
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 10
        if hasattr(scene.cycles, "use_fast_gi"):
            scene.cycles.use_fast_gi = True
        scene.cycles.device = "GPU"
        if hasattr(scene.cycles, "preview_samples"):
            scene.cycles.preview_samples = max(8, int(args.samples) // 2)
        if hasattr(bpy.context.view_layer, "cycles"):
            bpy.context.view_layer.cycles.use_denoising = True
            if hasattr(bpy.context.view_layer.cycles, "denoising_store_passes"):
                bpy.context.view_layer.cycles.denoising_store_passes = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
        except Exception:
            scene.cycles.device = "CPU"
    else:
        if hasattr(scene, "eevee"):
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = int(args.samples)
            if hasattr(scene.eevee, "use_bloom"):
                scene.eevee.use_bloom = False
            if hasattr(scene.eevee, "use_gtao"):
                scene.eevee.use_gtao = True
            if hasattr(scene.eevee, "use_motion_blur"):
                scene.eevee.use_motion_blur = False
            if hasattr(scene.eevee, "shadow_cube_size"):
                scene.eevee.shadow_cube_size = "4096"
            if hasattr(scene.eevee, "shadow_cascade_size"):
                scene.eevee.shadow_cascade_size = "4096"
            if hasattr(scene.eevee, "use_ssr"):
                scene.eevee.use_ssr = True
            if hasattr(scene.eevee, "use_ssr_refraction"):
                scene.eevee.use_ssr_refraction = True
            if hasattr(scene.eevee, "use_volumetric_lights"):
                scene.eevee.use_volumetric_lights = False


def frame_target(trajectory: list[dict], frame_number: int) -> Vector:
    row = trajectory[min(max(frame_number - 1, 0), len(trajectory) - 1)]
    return Vector((float(row["x"]), float(row["y"]), 0.72))


def shot_for_frame(shots: list[dict], frame_number: int) -> dict:
    for shot in shots:
        start_frame = int(shot["start"]) + 1
        end_frame = max(start_frame, int(shot["end"]))
        if start_frame <= frame_number <= end_frame:
            return shot
    return shots[-1]


def build_trackside_anchor(trajectory: list[dict], shot: dict) -> tuple[Vector, Vector]:
    start = min(max(int(shot["start"]), 0), len(trajectory) - 2)
    end = min(max(int(shot["end"]) - 1, start + 1), len(trajectory) - 1)
    mid = (start + end) // 2
    current = trajectory[mid]
    nxt = trajectory[min(mid + 3, len(trajectory) - 1)]
    p0 = Vector((float(current["x"]), float(current["y"]), 0.0))
    p1 = Vector((float(nxt["x"]), float(nxt["y"]), 0.0))
    tangent = p1 - p0
    if tangent.length < 1e-6:
        tangent = Vector((1.0, 0.0, 0.0))
    tangent.normalize()
    normal = Vector((-tangent.y, tangent.x, 0.0))
    camera_position = p0 + normal * 12.0 - tangent * 4.0 + Vector((0.0, 0.0, 4.5))
    target = p0 + Vector((0.0, 0.0, 1.2))
    return camera_position, target


def animate_camera(scene: bpy.types.Scene, trajectory: list[dict], shots: list[dict]) -> None:
    bpy.ops.object.camera_add(location=(0.0, -6.0, 2.0))
    camera = bpy.context.active_object
    camera.name = "ReplayCamera"
    camera.data.lens = 33.0
    camera.data.sensor_width = 36.0
    camera.data.dof.use_dof = False
    scene.camera = camera

    focus_target = bpy.data.objects.new("FocusTarget", None)
    bpy.context.scene.collection.objects.link(focus_target)
    camera.data.dof.focus_object = focus_target

    smoothed_location: Vector | None = None
    smoothed_target: Vector | None = None
    for frame_number in range(1, len(trajectory) + 1):
        car_target = frame_target(trajectory, frame_number)
        car_yaw = float(trajectory[frame_number - 1].get("yaw", 0.0))
        car_speed = float(trajectory[frame_number - 1].get("speed", 18.0))
        chase_distance = 6.8 + min(max(car_speed - 20.0, 0.0) * 0.045, 1.4)
        side_bias = 0.65
        desired_location = car_target + Vector(
            (
                -chase_distance * math.cos(car_yaw) - side_bias * math.sin(car_yaw),
                -chase_distance * math.sin(car_yaw) + side_bias * math.cos(car_yaw),
                1.65,
            )
        )
        desired_target = car_target + Vector((20.0 * math.cos(car_yaw), 20.0 * math.sin(car_yaw), 0.58))
        camera.data.lens = 31.0

        if smoothed_location is None:
            smoothed_location = desired_location
        else:
            smoothed_location = smoothed_location.lerp(desired_location, 0.18)
        if smoothed_target is None:
            smoothed_target = desired_target
        else:
            smoothed_target = smoothed_target.lerp(desired_target, 0.22)

        camera_location = smoothed_location
        car_target = smoothed_target
        camera.location = camera_location
        camera.rotation_euler = look_at_rotation(camera_location, car_target)
        focus_target.location = car_target
        camera.keyframe_insert(data_path="location", frame=frame_number)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_number)
        focus_target.keyframe_insert(data_path="location", frame=frame_number)
        camera.data.keyframe_insert(data_path="lens", frame=frame_number)


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    frames_dir = Path(args.frames_dir)
    manifest = load_json(bundle_dir / "scene_manifest.json")
    camera_script = load_json(bundle_dir / "camera_script.json")
    trajectory = load_json(bundle_dir / "trajectory.json")
    if args.frame_limit is not None:
        trajectory = trajectory[: max(2, int(args.frame_limit))]

    clear_scene()
    scene = bpy.context.scene

    track_points = read_centerline(bundle_dir / manifest["track_csv"])
    add_ground()
    build_track(track_points, float(manifest["track"]["width"]), bool(manifest["track"].get("closed", True)))
    vehicle_asset = Path(args.vehicle_asset) if args.vehicle_asset else None
    vehicle = create_vehicle(manifest["vehicle"], vehicle_asset=vehicle_asset)
    animate_vehicle(vehicle, trajectory)
    if any(bool(row.get("planner_debug")) for row in trajectory):
        create_planner_debug_overlays(trajectory)
    add_lights()
    animate_camera(scene, trajectory, camera_script["shots"])
    configure_render(
        scene,
        args,
        fps=int(manifest.get("fps", 20)),
        frame_end=len(trajectory),
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")
    if args.save_blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend_path))
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
