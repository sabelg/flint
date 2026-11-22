"""Flint Studio take editor for existing SOMA30 BVH clips."""
from std.ffi import external_call
from std.sys import argv
from editor import draw_pose
from workflow_templates import labels, new_template, absolute
from flint.studio.source_editor import EmbeddedIde
from recipe_io import num, field, string, relative
from character import CharacterPreview, character_sheet
from flint.assets.glb_take import append_take_glb
from flint.platform.os import env_now
from transition import join_takes, model_path
from flint.ml.motionbricks import MbClip, load_bvh
from flint.ml.take_edit import edit_take, find_cycle, validate_take
from flint.platform.input import InputState
from flint.platform.os import (
    file_size,
    get_time,
    list_directory,
    make_dirs,
    remove_file,
    read_bytes,
    write_file_durable,
)
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import SimdCanvas


from take_files import load_take, save_new
from recipe import (
    contacts,
    Contact,
    export_partner,
    run_recipe,
    document,
    character_path,
    bone_map,
    partner_path,
    draw_pair,
    export_character,
)


def button(
    mut canvas: SimdCanvas,
    x: Int,
    y: Int,
    w: Int,
    label: String,
    active: Bool = False,
):
    canvas.draw_rect(x, y, w, 30, UInt32(0xFF286862 if active else 0xFF304557))
    canvas.draw_text(x + 9, y + 10, label, 0xEFF5F5, 1)


def project_takes() -> List[String]:
    var paths = List[String]()
    var folders: List[String] = [
        "build/kimodo-takes",
        "build/motion-recipes",
        "build/motion-takes",
        "build/kimodo",
        ".",
    ]
    for folder in folders:
        for name in list_directory(folder):
            if name.lower().endswith(".bvh") or name.lower().endswith(
                ".motionrecipe.json"
            ):
                paths.append(String(folder) + "/" + name)
    return paths^


def output_path(stem: String, extension: String = ".bvh") raises -> String:
    if not make_dirs("build/motion-takes"):
        raise Error("Could not create build/motion-takes")
    var n = 1
    var prefix = "build/motion-takes/" + stem + "-"
    while file_size(prefix + String(n) + extension) >= 0:
        n += 1
    return prefix + String(n) + extension


def run_editor(initial: String) raises:
    var win = WaylandWindow()
    win.set_title("Flint Studio - Motion takes")
    if not win.initialize(900, 560):
        raise Error(
            "Motion takes needs a display. Use --export for headless editing."
        )
    var canvas = SimdCanvas(900, 560)
    var viewport = SimdCanvas(900, 357)
    var input = InputState()
    var paths = project_takes()
    var path = initial.copy()
    var index = -1
    if not path and len(paths):
        index = 0
        path = paths[0]
    var source = MbClip()
    var preview = MbClip()
    var first = 0
    var last = 0
    var mirror = False
    var in_place = False
    var reverse = False
    var started = get_time()
    var status = String("Type a SOMA30 BVH path or generate a take in Kimodo.")
    var editing = not path
    var selected = editing
    var opening = Bool(path)
    var start_take = MbClip()
    var worker = Int32(-1)
    var recipe_job = False
    var opening_result = False
    var recipe_file = String()
    var staged_take = False
    var pending_recipe = String()
    var pending_rig = String()
    var pending_aliases = Dict[String, String]()
    var pending_tolerance: Float32 = 0
    var actor_pins = List[Contact]()
    var pending_pins = List[Contact]()
    var partner_rig = String()
    var partner_aliases = Dict[String, String]()
    var pending_partner_rig = String()
    var pending_partner_aliases = Dict[String, String]()
    var partner = MbClip()
    var partner_preview = MbClip()
    var paused = False
    var paused_frame = 0
    var library = False
    var ide = EmbeddedIde()
    var ide_env = String("FLINT_STUDIO_IDE_FILE")
    var aliases = Dict[String, String]()
    var tolerance: Float32 = 0
    var joined_path = String()
    var source_path = String()
    var character = CharacterPreview()
    var show_character = False
    var rig = env_now("FLINT_TAKE_CHARACTER")
    if not rig:
        rig = "assets/mozu.glb"
    while not win.close_requested:
        input.begin_frame()
        win.poll(input)
        if ide.present(win, input, False):
            continue
        if input.key_pressed(256):
            if library:
                library = False
                continue
            if not editing:
                break
            editing = False
            path = source_path.copy()
        var click = input.mouse_pressed(0)
        var x = Int(input.last_mouse_x)
        var y = Int(input.last_mouse_y)
        if (
            click
            and worker <= 0
            and library
            and y >= 73
            and y < 313
            and x >= 32
            and x < 868
        ):
            var kind = (y - 73) // 40 * 2 + Int(x >= 450)
            try:
                if not preview.num_frames():
                    raise Error("Open a SOMA take before creating a recipe")
                var current = source_path.copy()
                if (
                    mirror
                    or in_place
                    or reverse
                    or first
                    or last + 1 != source.num_frames()
                ):
                    current = output_path("recipe-source")
                    save_new(preview, current)
                var second = String()
                if start_take.num_frames():
                    second = output_path("recipe-partner")
                    save_new(start_take, second)
                path = new_template(kind, preview, current, second, rig)
                recipe_file = path.copy()
                var abs_file = absolute(path)
                _ = external_call["setenv", Int32](
                    ide_env.as_c_string_span(),
                    abs_file.as_c_string_span(),
                    Int32(1),
                )
                ide.close()
                ide.requested = True
                status = (
                    "Save the recipe in the editor, return here, then click"
                    " Open."
                )
            except err:
                status = String(err)
            library = False
            continue
        if click and y >= 14 and y < 44:
            if x >= 16 and x < 623:
                editing = True
                selected = True
            elif x >= 635 and x < 710:
                opening = True
            elif x >= 720 and x < 882:
                paths = project_takes()
                if len(paths):
                    index = (index + 1) % len(paths)
                    path = paths[index]
                    opening = True
                else:
                    status = (
                        "No project takes found. Enter a BVH path and click"
                        " Open."
                    )
        if editing and not opening:
            if input.key_pressed(257):
                opening = True
            elif input.key_pressed(259):
                path = String() if selected else String(
                    path[byte = 0 : max(0, path.byte_length() - 1)]
                )
                selected = False
            elif input.text_input_utf8:
                var chars = List[UInt8]()
                for b in input.text_input_utf8.as_bytes():
                    if b >= 32 and b < 127:
                        chars.append(b)
                if len(chars):
                    if selected:
                        path = ""
                    path += String(unsafe_from_utf8=chars)
                    selected = False
        var refresh = False
        if opening and worker > 0:
            opening = False
            status = "Wait for the current motion job."
        if (
            opening
            and path.lower().endswith(".motionrecipe.json")
            and not opening_result
        ):
            opening = False
            if worker > 0:
                status = "Wait for the current motion job."
            else:
                try:
                    var d = document(path)
                    pending_rig = character_path(d, path)
                    pending_aliases = bone_map(d)
                    pending_tolerance = num(d, d.root, "tolerance", 0)
                    if pending_tolerance < 0:
                        raise Error(
                            "Optimization tolerance must be nonnegative"
                        )
                    var actor = field(d, d.root, "partner")
                    pending_pins = contacts(d, 100000000)
                    var partner_character = string(d, actor, "character")
                    pending_partner_rig = relative(
                        path, partner_character
                    ) if partner_character else pending_rig.copy()
                    pending_partner_aliases = (
                        bone_map(d, actor) if partner_character
                        or field(d, actor, "bones")
                        >= 0 else pending_aliases.copy()
                    )
                    joined_path = output_path(
                        "recipe-" + String(external_call["getpid", Int32]())
                    )
                    pending_recipe = path.copy()
                    worker = external_call["fork", Int32]()
                    if worker == 0:
                        _ = external_call["close_range", Int32](
                            UInt32(3), UInt32(0xFFFFFFFF), UInt32(0)
                        )
                        try:
                            run_recipe(path, joined_path, "")
                        except err:
                            var message = String(err)
                            var error_bytes = List[UInt8](message.as_bytes())
                            _ = write_file_durable(
                                joined_path + ".error",
                                error_bytes,
                            )
                            print("Motion recipe:", err, flush=True)
                            external_call["_exit", NoneType](Int32(1))
                        external_call["_exit", NoneType](Int32(0))
                    recipe_job = worker > 0
                    status = (
                        "Baking recipe..." if recipe_job else "Could not launch recipe"
                    )
                    editing = False
                except err:
                    status = String(err)
        if opening:
            opening = False
            try:
                source = load_take(joined_path if opening_result else path)
                if opening_result:
                    partner = MbClip()
                    character = CharacterPreview()
                    show_character = False
                    if file_size(partner_path(joined_path)) >= 0:
                        partner = load_take(partner_path(joined_path))
                    staged_take = True
                    rig = pending_rig.copy()
                    aliases = pending_aliases.copy()
                    tolerance = pending_tolerance
                    actor_pins = pending_pins.copy()
                    partner_rig = pending_partner_rig.copy()
                    partner_aliases = pending_partner_aliases.copy()
                    recipe_file = pending_recipe.copy()
                    source_path = joined_path.copy()
                    path = recipe_file.copy()
                    opening_result = False
                else:
                    partner = MbClip()
                    staged_take = False
                    aliases = Dict[String, String]()
                    tolerance = 0
                    recipe_file = ""
                    source_path = path.copy()
                    rig = env_now("FLINT_TAKE_CHARACTER")
                    if not rig:
                        rig = "assets/mozu.glb"
                    character = CharacterPreview()
                    show_character = False
                first = 0
                last = source.num_frames() - 1
                mirror = False
                in_place = False
                reverse = False
                refresh = True
                editing = False
                status = "Loaded " + path
            except err:
                status = String(err)
                source = MbClip()
                preview = MbClip()
                source_path = ""
                editing = True
                selected = True
        if source.num_frames() and click and y >= 457 and y < 487:
            editing = False
            if x >= 16 and x < 124:
                mirror = not mirror
                refresh = True
            elif x >= 132 and x < 240:
                in_place = not in_place
                refresh = True
            elif x >= 248 and x < 356:
                reverse = not reverse
                refresh = True
            elif x >= 364 and x < 487:
                try:
                    var bounds = find_cycle(source, 0, source.num_frames() - 1)
                    first = bounds[0]
                    last = bounds[1]
                    refresh = True
                except err:
                    status = String(err)
            elif x >= 495 and x < 610:
                first = 0
                last = source.num_frames() - 1
                refresh = True
            elif x >= 750 and x < 882:
                try:
                    var parts = source_path.split("/")
                    var base = String(parts[len(parts) - 1])
                    var stem = String(
                        base[byte = 0 : base.byte_length() - 4]
                    ) if base.lower().endswith(".bvh") else base.copy()
                    var dest = output_path(stem)
                    save_new(preview, dest)
                    if partner_preview.num_frames():
                        try:
                            save_new(partner_preview, partner_path(dest))
                        except err:
                            _ = remove_file(dest)
                            raise Error(String(err))
                    status = "Saved " + dest
                except err:
                    status = String(err)
        if worker > 0:
            var code = Int32(0)
            if (
                external_call["waitpid", Int32](
                    worker, Pointer(to=code), Int32(1)
                )
                == worker
            ):
                worker = -1
                if code == 0:
                    if recipe_job:
                        opening_result = True
                        path = pending_recipe.copy()
                    else:
                        path = joined_path.copy()
                    opening = True
                else:
                    status = "Motion job failed. See the Studio console."
                    if recipe_job and file_size(joined_path + ".error") > 0:
                        status = String(
                            unsafe_from_utf8=read_bytes(joined_path + ".error")
                        )
                        _ = remove_file(joined_path + ".error")
                recipe_job = False
        elif preview.num_frames() and click and y >= 498 and y < 528:
            if x >= 16 and x < 166:
                start_take = preview.copy()
                status = (
                    "Start pose set. Open the destination take, then click Join"
                    " to this take."
                )
            elif x >= 174 and x < 348:
                try:
                    if not start_take.num_frames():
                        raise Error("Choose Use as start on the first take")
                    if file_size(model_path()) < 1000:
                        raise Error(
                            "Set FLINT_MOTIONBRICKS_MODEL to the SOMA fp32 ONNX"
                            " file"
                        )
                    joined_path = output_path("transition")
                    worker = external_call["fork", Int32]()
                    if worker == 0:
                        _ = external_call["close_range", Int32](
                            UInt32(3), UInt32(0xFFFFFFFF), UInt32(0)
                        )
                        try:
                            var joined = join_takes(start_take, preview)
                            save_new(joined, joined_path)
                        except err:
                            print("MotionBricks:", err, flush=True)
                            external_call["_exit", NoneType](Int32(1))
                        external_call["_exit", NoneType](Int32(0))
                    status = (
                        "Generating transition..." if worker
                        > 0 else "Could not launch MotionBricks"
                    )
                except err:
                    status = String(err)
        if preview.num_frames() and click and y >= 498 and y < 528:
            if x >= 360 and x < 530:
                if show_character:
                    show_character = False
                else:
                    try:
                        if not character.model.skins:
                            character.load(rig)
                        character.bake(preview, aliases, staged_take)
                        show_character = True
                        status = "Previewing " + rig
                    except err:
                        status = String(err)
            elif x >= 540 and x < 706 and show_character:
                try:
                    var dest = output_path("character-take", ".glb")
                    export_character(
                        preview, rig, aliases, tolerance, dest, "", staged_take
                    )
                    if partner_preview.num_frames():
                        try:
                            export_partner(
                                partner_preview,
                                dest,
                                partner_rig,
                                partner_aliases,
                                aliases,
                                actor_pins,
                                tolerance,
                                first,
                                mirror,
                                reverse,
                            )
                        except err:
                            _ = remove_file(dest)
                            raise Error(String(err))
                    status = "Saved " + dest
                except err:
                    status = String(err)
        if click and y >= 498 and y < 528 and x >= 714 and x < 880:
            library = not library
            editing = False
        if (
            click
            and y >= 70
            and y < 100
            and x >= 700
            and not library
            and recipe_file
        ):
            var abs_file = absolute(recipe_file)
            _ = external_call["setenv", Int32](
                ide_env.as_c_string_span(),
                abs_file.as_c_string_span(),
                Int32(1),
            )
            ide.close()
            ide.requested = True
        if refresh:
            paused = False
            if partner.num_frames():
                partner_preview = edit_take(
                    partner, first, last, mirror, in_place, reverse
                )
            else:
                partner_preview = MbClip()
            preview = edit_take(source, first, last, mirror, in_place, reverse)
            if show_character:
                try:
                    character.bake(preview, aliases, staged_take)
                except err:
                    show_character = False
                    status = String(err)
            started = get_time()
        canvas.clear()
        var frame = 0
        if preview.num_frames():
            frame = (
                Int((get_time() - started) * Float64(preview.sample_rate))
                % preview.num_frames()
            )
            if not editing and input.key_pressed(32):
                if paused:
                    started = get_time() - Float64(paused_frame) / Float64(
                        preview.sample_rate
                    )
                else:
                    paused_frame = frame
                paused = not paused
            if click and y >= 420 and y < 451 and x >= 490 and x < 880:
                paused_frame = min(
                    preview.num_frames() - 1,
                    (x - 490) * preview.num_frames() // 390,
                )
                paused = True
            if paused:
                frame = paused_frame
            viewport.clear()
            if show_character:
                character.draw(viewport, Float32(frame) / preview.sample_rate)
            elif partner_preview.num_frames():
                draw_pair(viewport, preview, partner_preview, frame)
            else:
                draw_pose(viewport, preview, frame)
            for i in range(len(viewport.pixels)):
                canvas.pixels[56 * 900 + i] = viewport.pixels[i]
        canvas.draw_rect(0, 0, 900, 56, 0xFF16232F)
        canvas.draw_rect(
            16, 14, 607, 30, UInt32(0xFF35506A if editing else 0xFF213342)
        )
        canvas.draw_text(
            24,
            24,
            String(
                path[byte = max(0, path.byte_length() - 70) :]
            ) if path else String("SOMA30 BVH path"),
            0xEFF5F5,
            1,
        )
        button(canvas, 635, 14, 75, "Open")
        button(canvas, 720, 14, 162, "Next project take")
        canvas.draw_rect(0, 413, 900, 147, 0xFF16232F)
        if preview.num_frames():
            canvas.draw_text(
                16,
                431,
                "Source frames "
                + String(first + 1)
                + " to "
                + String(last + 1)
                + "     Preview "
                + String(frame + 1)
                + " / "
                + String(preview.num_frames()),
                0xE2E9EB,
                1,
            )
        button(canvas, 16, 457, 108, "Mirror L/R", mirror)
        button(canvas, 132, 457, 108, "In place", in_place)
        button(canvas, 248, 457, 108, "Reverse", reverse)
        button(
            canvas,
            364,
            457,
            123,
            "Find loop",
            source.num_frames() > 0 and last - first + 1 < source.num_frames(),
        )
        button(canvas, 495, 457, 115, "Full range")
        button(canvas, 750, 457, 132, "Save new BVH")
        button(
            canvas, 16, 498, 150, "Use as start", start_take.num_frames() > 0
        )
        button(canvas, 174, 498, 174, "Join to this take", worker > 0)
        button(canvas, 360, 498, 170, "Character preview", show_character)
        if show_character:
            button(canvas, 540, 498, 166, "Save character GLB")
        button(canvas, 714, 498, 166, "Author tools", library)
        if preview.num_frames():
            canvas.draw_rect(490, 432, 390, 3, 0xFF506677)
            canvas.draw_rect(
                490 + frame * 382 // max(1, preview.num_frames() - 1),
                426,
                8,
                14,
                0xFF81CDC1,
            )
        if library:
            canvas.draw_rect(22, 63, 856, 258, 0xFF152535)
            var tools = labels()
            for i in range(len(tools)):
                button(
                    canvas,
                    32 + (i % 2) * 418,
                    73 + (i // 2) * 40,
                    408,
                    tools[i],
                )
        elif recipe_file:
            button(canvas, 700, 70, 180, "Edit recipe")
        canvas.draw_text(16, 541, status, 0xD0DCDD, 1)
        win.present(canvas.pixels, 900, 560)
        _ = external_call["usleep", Int32](UInt32(16000))
    if worker > 0:
        _ = external_call["kill", Int32](worker, Int32(15))
        var code = Int32(0)
        _ = external_call["waitpid", Int32](worker, Pointer(to=code), Int32(0))
    ide.close()
    win.close()


def main() raises:
    var args = argv()
    if len(args) == 2 and (args[1] == "--help" or args[1] == "-h"):
        print("studio --takes [input.bvh or file.motionrecipe.json]")
        print(
            "studio --takes --recipe file.motionrecipe.json new.bvh|new.glb"
            " [preview.png]"
        )
        print(
            "studio --takes --export input.bvh new.bvh [--loop] [--mirror]"
            " [--in-place] [--reverse]"
        )
        print("studio --takes --join start.bvh target.bvh new.bvh")
        print(
            "studio --takes --retarget input.bvh character.glb new.glb"
            " [preview.png]"
        )
        return
    if len(args) >= 2 and args[1] == "--recipe":
        if len(args) != 4 and len(args) != 5:
            raise Error(
                "Usage: studio --takes --recipe file.motionrecipe.json"
                " new.bvh|new.glb [preview.png]"
            )
        run_recipe(
            String(args[2]),
            String(args[3]),
            String(args[4]) if len(args) == 5 else String(),
        )
        return
    if len(args) >= 2 and args[1] == "--retarget":
        if len(args) != 5 and len(args) != 6:
            raise Error(
                "Usage: studio --takes --retarget input.bvh character.glb"
                " new.glb [preview.png]"
            )
        var take = load_take(String(args[2]))
        var character = CharacterPreview()
        character.load(String(args[3]))
        character.bake(take)
        append_take_glb(
            String(args[3]),
            String(args[4]),
            character.model.animations[character.clip],
        )
        if len(args) == 6:
            character_sheet(character, take, String(args[5]))
        print("Saved", take.num_frames(), "retargeted frames to", args[4])
        return
    if len(args) >= 2 and args[1] == "--join":
        if len(args) != 5:
            raise Error(
                "Usage: studio --takes --join start.bvh target.bvh new.bvh"
            )
        var joined = join_takes(
            load_take(String(args[2])), load_take(String(args[3]))
        )
        save_new(joined, String(args[4]))
        print("Saved", joined.num_frames(), "frames to", args[4])
        return
    if len(args) < 3 and (len(args) == 1 or args[1] != "--export"):
        run_editor(String(args[1]) if len(args) == 2 else String())
        return
    if len(args) < 4 or args[1] != "--export":
        raise Error(
            "Usage: studio --takes [input.bvh] | --export input.bvh new.bvh"
            " [--loop] [--mirror] [--in-place] [--reverse]"
        )
    var source = load_take(String(args[2]))
    var first = 0
    var last = source.num_frames() - 1
    var mirror = False
    var in_place = False
    var reverse = False
    for i in range(4, len(args)):
        if args[i] == "--loop":
            var bounds = find_cycle(source, 0, source.num_frames() - 1)
            first = bounds[0]
            last = bounds[1]
        elif args[i] == "--mirror":
            mirror = True
        elif args[i] == "--in-place":
            in_place = True
        elif args[i] == "--reverse":
            reverse = True
        else:
            raise Error("Unknown take edit " + String(args[i]))
    var result = edit_take(source, first, last, mirror, in_place, reverse)
    save_new(result, String(args[3]))
    print(
        "Exported source frames", first + 1, "through", last + 1, "to", args[3]
    )
