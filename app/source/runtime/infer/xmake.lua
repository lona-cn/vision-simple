local target_name = "infer"
local kind = "object"
local group_name = "runtime"
local pkgs = { "opencv", "magic_enum" }
if has_config("with_dml") then
    table.insert(pkgs, "directml")
    table.insert(pkgs, "onnxruntime-dml")
else
    table.insert(pkgs, "onnxruntime")
end
local deps = { "common" }
local syslinks = {}
if is_os("windows") then
    table.insert(syslinks, "User32")
    table.insert(syslinks, "dxgi")
end
local function callback()
    set_basename("vs" .. target_name .. "_$(plat)_$(arch)_$(mode)")
    add_options("with_dml", "with_cuda", "with_tensorrt")
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)