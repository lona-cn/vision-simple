local target_name = "server"
local kind = "binary"
local group_name = "program"
local pkgs = { "libhv", "turbobase64" }
local deps = { "runtime", "infer" }
local syslinks = {}
local function callback()
    add_extrafiles(path.join(os.projectdir(), "doc", "openapi", "*"))
    --add_ldflags("-Wl,-Bdynamic","-lonnxruntime_providers_shared","-lonnxruntime","-Wl,-Bstatic", "-static-libgcc", {force = true})
    --add_ldflags("/usr/aarch64-linux-gnu/lib/libc.a","/usr/aarch64-linux-gnu/lib/libm.a", {force = true})
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)