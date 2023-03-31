import bpy

destfolder = "C:\\"

sce = bpy.data.scenes['Scene']
for ob in sce.objects:
    bpy.ops.object.select_pattern(pattern = ob.name)
    bpy.ops.export_scene.obj(filepath = destfolder + ob.name + ".obj", use_selection = True)