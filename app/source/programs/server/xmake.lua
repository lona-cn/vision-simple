local target_name = "server"
local kind = "binary"
local group_name = "program"
local pkgs = { "libhv", "turbobase64" }
local deps = { "runtime", "infer" }
local syslinks = {}
local function callback()
    add_extrafiles(path.join(os.projectdir(), "doc", "openapi", "*"))
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)