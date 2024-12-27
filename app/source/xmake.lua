add_requires("openmp","magic_enum 0.9.6")
if has_config("with_dml") then
add_requires("directml", { system = false })
add_requires("onnxruntime-dml", { system = false })
else
add_requires("onnxruntime", { system = false } )
end
if is_os("windows") then
add_requires("opencv 4.10.0")
add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
protobuf = false } })
elseif is_os("linux") then
add_requires("opencv",{ system = true })
if has_config("with_cuda") then
add_requireconfs("onnxruntime", { system = false, configs = { gpu = true } })
end
end

includes("third/xmake.lua")
includes("runtime/xmake.lua")
includes("programs/xmake.lua")