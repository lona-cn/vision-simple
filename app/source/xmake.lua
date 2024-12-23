add_requires("openmp","onnxruntime","opencv 4.10.0","magic_enum 0.9.6")
--add_requireconfs("onnxruntime-dml", { system = false, configs = { dml = true } })
add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
protobuf = false } })

includes("third/xmake.lua")
includes("runtime/xmake.lua")
includes("programs/xmake.lua")