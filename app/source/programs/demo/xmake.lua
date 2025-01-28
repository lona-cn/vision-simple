local target_name = "demo"
local kind = "binary"
local group_name = "program"
local pkgs = {}
local deps = {"runtime"}
local syslinks = {}
local function callback()
    set_basename("vision_simple-demo")
end
CreateTarget(target_name,kind,os.scriptdir(),group_name,pkgs,deps,syslinks,callback)