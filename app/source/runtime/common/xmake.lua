local target_name = "common"
local kind = "static"
local group_name = "runtime"
local pkgs = {"yalantinglibs","magic_enum"}
local deps = {}
local syslinks = {}
local function callback()
end
CreateTarget(target_name,kind,os.scriptdir(),group_name,pkgs,deps,syslinks,callback)