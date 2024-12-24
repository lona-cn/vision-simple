#pragma once
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#include <dxgi.h>
#include <wrl.h>
#endif

namespace vision_simple
{
    struct DXAdapterInfo
    {
        uint32_t index;
        std::wstring description;
        double vram_in_MBytes;
        double vram_in_GBytes;

        bool IsBasicRenderDriver() const
        {
            return this->description == L"Microsoft Basic Render Driver";
        }

        static std::vector<DXAdapterInfo> ListAdapters() noexcept
        {
#ifndef _WIN32
        return std::vector<DXAdapterInfo>{};
#else
            std::vector<DXAdapterInfo> results;
            // 创建 IDXGIFactory 对象
            Microsoft::WRL::ComPtr<IDXGIFactory> factory;
            HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                           reinterpret_cast<void**>(factory.
                                               GetAddressOf()));
            if (FAILED(hr))
            {
                std::cerr << "Failed to create DXGIFactory" << std::endl;
                return {};
            }

            // 枚举所有适配器
            UINT i = 0;
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            while (factory->EnumAdapters(i, adapter.ReleaseAndGetAddressOf()) !=
                DXGI_ERROR_NOT_FOUND)
            {
                DXGI_ADAPTER_DESC desc;
                adapter->GetDesc(&desc);
                results.emplace_back(i, desc.Description,
                                     desc.DedicatedVideoMemory / (1024.0 * 1024),
                                     desc.DedicatedVideoMemory / (1024.0 * 1024 * 1024));
#if defined(VISION_SIMPLE_DEBUG)
                // 打印适配器的基本信息
                std::wcout << L"Adapter " << i << L":" << std::endl;
                std::wcout << L"  Description: " << desc.Description << std::endl;
                std::wcout << L"  Vendor ID: " << desc.VendorId << std::endl;
                std::wcout << L"  Device ID: " << desc.DeviceId << std::endl;
                std::wcout << L"  SubSys ID: " << desc.SubSysId << std::endl;
                std::wcout << L"  Revision: " << desc.Revision << std::endl;
                std::wcout << L"  Dedicated Video Memory: " << desc.DedicatedVideoMemory / (
                        1024 * 1024) << L" MB" <<
                    std::endl;
                std::wcout << L"  Dedicated System Memory: " << desc.DedicatedSystemMemory /
                    (1024 * 1024) << L" MB" <<
                    std::endl;
                std::wcout << L"  Shared System Memory: " << desc.SharedSystemMemory / (
                    1024 * 1024) << L" MB" << std::endl;
                std::wcout << std::endl;
#endif
                i++;
            }
            return results;
#endif
        }
    };
}
