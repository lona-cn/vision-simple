local target_name = "server"
local kind = "static"
local group_name = "runtime"
local pkgs = {"libhv"}
local deps = {"common","config"}
local syslinks = {}
local function callback()
	set_basename("vs_server_$(arch)_$(mode)")
end
CreateTarget(target_name,kind,os.scriptdir(),group_name,pkgs,deps,syslinks,callback)