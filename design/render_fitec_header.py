"""Render FITEC's header artwork with Blender's embedded Python.

From the repository root, run Blender with
`--background --factory-startup --python design/render_fitec_header.py`.
"""

from math import exp
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
RANGE = (-3.5, 3.5)


def material(name, color):
    def linear(channel):
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*[linear(int(color[i:i + 2], 16) / 255) for i in (0, 2, 4)], 1)
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mat.diffuse_color = emission.inputs["Color"].default_value
    return mat


def line(name, points, stroke, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = stroke
    curve.bevel_resolution = 2
    curve.resolution_u = 1
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for vertex, (x, y, z) in zip(spline.points, points):
        vertex.co = (x, y, z, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)


def sample(start, stop, count):
    return [start + i * (stop - start) / (count - 1) for i in range(count)]


def height(x, y):
    return 2.35 * exp(-0.5 * ((x / 1.08) ** 2 + (y / 0.87) ** 2))


def create_artwork():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    grid = material("Petroleum grid", "224C5D")
    fine_paper = material("Low contour", "39838D")
    surface = material("Turquoise contour", "438F96")
    ridge = material("Highlighted normal curve", "83DCCC")
    dots = material("Reference points", "B4F4E2")

    for x in sample(*RANGE, 15):
        line("Grid X", [(x, -1.6, 0), (x, 1.6, 0)], 0.012, grid)
    for y in sample(-1.6, 1.6, 9):
        line("Grid Y", [(RANGE[0], y, 0), (RANGE[1], y, 0)], 0.012, grid)

    for y in (-1.55, -1.15, -0.75, -0.38, 0.38, 0.75, 1.15, 1.55):
        line("Distribution section",
             [(x, y, height(x, y)) for x in sample(*RANGE, 101)],
             0.017, fine_paper if abs(y) > 0.8 else surface)
    for x in sample(-2.7, 2.7, 17):
        line("Distribution cross-section",
             [(x, y, height(x, y)) for y in sample(-1.55, 1.55, 65)],
             0.012, fine_paper)
    line("Normal curve", [(x, 0, height(x, 0)) for x in sample(*RANGE, 121)], 0.037, ridge)

    for x in (-1.08, 0, 1.08):
        line("Sigma projection", [(x, 0, 0), (x, 0, height(x, 0))], 0.009, surface)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.065, location=(x, 0, height(x, 0)))
        marker = bpy.context.object
        marker.name = "Normal curve marker"
        marker.data.materials.append(dots)

    center = Vector((-1.05, 0, 0.80))
    bpy.ops.object.camera_add(location=center + Vector((2.9, -10.5, 4.6)))
    camera = bpy.context.object
    camera.name = "Header camera"
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 13.8
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 176
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = str(ROOT / "probviz" / "assets" / "fitec-blender.png")

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.overlay.show_overlays = False
                area.spaces.active.shading.color_type = "MATERIAL"
    bpy.ops.object.select_all(action="DESELECT")
    source = ROOT / "build" / "fitec-header-design.blend"
    source.parent.mkdir(exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(source), compress=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    create_artwork()
