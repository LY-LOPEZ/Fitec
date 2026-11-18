"""Render the four transparent FITEC symbols using Blender in background mode."""

from math import cos, exp, pi, radians, sin
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_fitec_header import ROOT, line, material


def draw_symbol(name, white, mint, ink):
    def stroke(points, width=.045, color=white):
        line(name, [(x, y, .1) for x, y in points], width, color)

    def dot(x, y, radius, color):
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=.025,
                                            location=(x, y, .22))
        bpy.context.object.data.materials.append(color)

    if name == "brand":
        angle = radians(16)
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        obj.dimensions = (1.55, 1.55, .3)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.rotation_euler.z = angle
        bevel = obj.modifiers.new("Rounded corners", "BEVEL")
        bevel.width, bevel.segments = .2, 8
        obj.data.materials.append(white)
        for x, y in ((0, 0), (-.42, -.42), (.42, -.42), (-.42, .42), (.42, .42)):
            dot(x*cos(angle)-y*sin(angle), x*sin(angle)+y*cos(angle), .14, ink)
    elif name == "model":
        stroke([(-.94 + i*.0188, -.52 + 1.35*exp(-((-.94+i*.0188)/.43)**2))
                for i in range(101)], .065)
        stroke([(-.98, -.65), (.98, -.65)], .04, mint)
    elif name == "dice":
        stroke([(0,.9), (.82,.46), (.82,-.46), (0,-.9), (-.82,-.46),
                (-.82,.46), (0,.9)])
        stroke([(-.82,.46), (0,0), (.82,.46)], .04, mint)
        stroke([(0,0), (0,-.9)], .04, mint)
        for x, y in ((0,.48), (-.55,-.04), (-.28,-.39), (.39,-.28)):
            dot(x, y, .087, white)
    else:
        corners = ((.49,.36,0), (-.49,.36,90), (-.49,-.36,180), (.49,-.36,270))
        points = [(x+.19*cos(radians(start+i*9)), y+.19*sin(radians(start+i*9)))
                  for x, y, start in corners for i in range(11)]
        stroke(points + [points[0]], .065)
        stroke([(0,.55),(0,.82)], .05)
        dot(0,.85,.095,mint)
        for x in (-.27,.27):
            dot(x,.08,.095,white)
        stroke([(-.21,-.23),(.21,-.23)], .05, mint)
        for x in (-.82,.82):
            stroke([(x,-.2),(x,.18)], .06)


def render():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    bpy.ops.object.camera_add(location=(0,0,5), rotation=(0,0,0))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 2.12
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = 192
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    palette = [material(label, color) for label, color in
               (("Porcelain","F6FFFF"), ("Turquoise","86E4DC"), ("Petroleum","0B3743"))]
    for name in ("brand", "model", "dice", "assistant"):
        draw_symbol(name, *palette)
        scene.render.filepath = str(ROOT / "probviz" / "assets" / f"symbol-{name}.png")
        bpy.ops.render.render(write_still=True)
        for obj in list(scene.objects):
            if obj != scene.camera:
                bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    render()
