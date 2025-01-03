local target_name = "infer"
local kind = "static"
local group_name = "runtime"
local pkgs = {"openmp","opencv","magic_enum"}
if has_config("with_dml") then
	table.insert(pkgs,"directml")
	table.insert(pkgs,"onnxruntime-dml")
else
	table.insert(pkgs,"onnxruntime")
end
local deps = {"common","config"}
local syslinks = {}
if is_os("windows") then
	table.insert(syslinks,"User32")
	table.insert(syslinks,"dxgi")
end
local function callback()
	set_basename("vs_infer_$(arch)_$(mode)")
	add_options("with_dml","with_cuda","with_tensorrt")
end
CreateTarget(target_name,kind,os.scriptdir(),group_name,pkgs,deps,syslinks,callback)