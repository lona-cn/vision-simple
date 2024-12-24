add_requires("openmp","opencv 4.10.0","magic_enum 0.9.6")
if has_config("with_dml") then
add_requires("directml", { system = false })
add_requires("onnxruntime-dml", { system = false })
else
add_requires("onnxruntime", { system = false } )
end
add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
protobuf = false } })

includes("third/xmake.lua")
includes("runtime/xmake.lua")
includes("programs/xmake.lua")