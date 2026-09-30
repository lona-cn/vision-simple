#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "ImageCodec.h"

using namespace vision_simple;
namespace {
void Check(bool ok, const char* message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
std::string Base64(std::span<const uint8_t> bytes) {
  constexpr char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t n = (uint32_t(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out += digits[(n >> 18) & 63];
    out += digits[(n >> 12) & 63];
    out += i + 1 < bytes.size() ? digits[(n >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? digits[n & 63] : '=';
  }
  return out;
}
std::vector<uint8_t> Bytes(std::string_view text) {
  return {text.begin(), text.end()};
}
void RoundTrip(std::span<const uint8_t> bytes, size_t pixels, std::string_view format) {
  auto prepared = PrepareEncodedImage(Base64(bytes));
  Check(prepared && prepared->pixels == pixels,
        (std::string(format) + ": header predicts exact decoded pixels").c_str());
  Check(std::equal(bytes.begin(), bytes.end(), prepared->bytes.begin(), prepared->bytes.end()),
        (std::string(format) + ": prepared bytes preserve original payload").c_str());
  auto image = DecodeImageBytes(prepared->bytes);
  Check(image && image->total() == pixels && image->type() == CV_8UC3,
        (std::string(format) + ": prepared format decodes to expected BGR pixels").c_str());
}
void Formats() {
  cv::Mat source(2, 3, CV_8UC3, cv::Scalar(17, 83, 201));
  for (const char* extension : {".png", ".jpg", ".bmp", ".ppm", ".pam"}) {
    std::vector<uint8_t> bytes;
    Check(cv::imencode(extension, source, bytes), "encode supported image format");
    RoundTrip(bytes, 6, extension);
  }
  cv::Mat floating(2, 3, CV_32FC3, cv::Scalar(0.1, 0.3, 0.7));
  for (const char* extension : {".pfm", ".hdr"}) {
    std::vector<uint8_t> bytes;
    Check(cv::imencode(extension, floating, bytes), "encode floating-point format");
    RoundTrip(bytes, 6, extension);
  }
  for (const char* text : {"P1\n3 2\n0 1 0 1 0 1\n",
                          "P2\n# dimensions follow\n3 2\n255\n0 1 2 3 4 5\n",
                          "P3\n3 2\n255\n0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17\n"})
    RoundTrip(Bytes(text), 6, std::string_view(text, 2));
  auto pbm = Bytes("P4\n3 2\n");
  pbm.insert(pbm.end(), {0x40, 0xa0});
  RoundTrip(pbm, 6, "P4");
  auto pgm = Bytes("P5\n3 2\n255\n");
  pgm.insert(pgm.end(), {0, 1, 2, 3, 4, 5});
  RoundTrip(pgm, 6, "P5");
  auto pfm = Bytes("Pf\n3 2\n-1.0\n");
  pfm.resize(pfm.size() + 6 * sizeof(float), 0);
  auto grayscale = PrepareEncodedImage(Base64(pfm));
  Check(grayscale && grayscale->pixels == 6, "Pf: grayscale header preflight counts pixels");
  // OpenCV 4.10 PFMDecoder::readData uses convertTo, which preserves the source
  // channels rather than satisfying IMREAD_COLOR for Pf. Retain the existing
  // CV_8UC3 output contract: preflight acceptance is not decode acceptance.
  Check(!DecodeImageBytes(grayscale->bytes), "Pf: existing unsupported-output-type rejection");
  // Sun raster: uncompressed 24-bit, two padded scanlines of 3 BGR pixels.
  std::vector<uint8_t> ras(52, 0);
  auto put = [&](size_t at, uint32_t n) {
    for (size_t i = 0; i < 4; ++i) ras[at + i] = uint8_t(n >> (24 - 8 * i));
  };
  put(0, 0x59a66a95); put(4, 3); put(8, 2); put(12, 24); put(16, 20); put(20, 1);
  RoundTrip(ras, 6, "SunRAS");
  Check(!DecodeEncodedImage(Base64(pgm), 6), "legacy bounded decode remains PNG/JPEG only");
}
void BadHeadersAndLimits() {
  for (const char* text : {"", "P6\n3 2\n", "P6\n0 2\n255\n", "P6\n-1 2\n255\n",
                          "P6\n2147483648 2\n255\n", "P6\n999999999999999999999999 2\n255\n",
                          "P7\nWIDTH 3\nHEIGHT 2\nDEPTH 3\nMAXVAL 255\n",
                          "P7\nWIDTH 3\nWIDTH 4\nHEIGHT 2\nDEPTH 3\nMAXVAL 255\nENDHDR\n",
                          "Pf\n3 2\nnan\n", "#?RADIANCE\n\n-Y 2 +X 3\n"})
    Check(!PrepareEncodedImage(Base64(Bytes(text))), "malformed or unrepresentable header rejected");
  for (const char* text : {"A===", "AA=A", "AB==", "AAB=", "AAAA\n", "data:image/png;base64,AAAA"})
    Check(!PrepareEncodedImage(text), "noncanonical base64 rejected");
  auto huge = PrepareEncodedImage(Base64(Bytes("P6\n2147483647 2147483647\n255\n")));
  const uint64_t product = uint64_t(INT_MAX) * INT_MAX;
  if (product <= std::numeric_limits<size_t>::max())
    Check(huge && huge->pixels == product, "huge declaration counted without codec allocation");
  else
    Check(!huge, "pixel product overflow rejected");
  std::vector<uint8_t> png;
  Check(cv::imencode(".png", cv::Mat(2, 3, CV_8UC3, cv::Scalar()), png), "encode PNG boundary fixture");
  auto broken = png;
  broken[16] = 0x80;
  Check(!PrepareEncodedImage(Base64(broken)), "PNG dimensions exceeding INT_MAX rejected");
  for (size_t n = 0; n < 33; ++n)
    Check(!PrepareEncodedImage(Base64(std::span(png).first(n))), "truncated PNG header rejected");
  Check(!DecodeEncodedImage(Base64(png), 5), "legacy pixel limit rejects exact overage");
  Check(DecodeEncodedImage(Base64(png), 6).has_value(), "legacy pixel limit admits exact boundary");
}
void BinaryHeaderBoundaries() {
  // A number-adjacent comment marker is consumed differently by OpenCV PNM
  // ReadNumber: comment digits become the next dimension, bypassing reservation.
  auto ambiguous = Bytes("P4\n1#4\n1\n");
  ambiguous.insert(ambiguous.end(), {0, 0, 0, 0});
  Check(!PrepareEncodedImage(Base64(ambiguous)),
        "P4: reject number-adjacent comment delimiter before codec allocation");
  auto commented = Bytes("P4\n1 #4\n4\n");
  commented.insert(commented.end(), {0, 0, 0, 0});
  RoundTrip(commented, 4, "P4 whitespace-delimited comment");
  for (const char* header : {"P6\n# comment\r3 2\n255\n",
                             "P6\r\n3 # comment\r\n2\r\n255\n"}) {
    auto cr_comment = Bytes(header);
    cr_comment.resize(cr_comment.size() + 18, 42);
    RoundTrip(cr_comment, 6, "P6 CR/CRLF-delimited comment");
  }
  for (const char* header : {"P4\n1 1#4\n", "P6\n1 1\n255#4\n"})
    Check(!PrepareEncodedImage(Base64(Bytes(header))),
          "PNM: reject number-adjacent comment in every numeric header field");
  auto pam = Bytes("P7\rWIDTH 3\rHEIGHT 2\rDEPTH 3\rMAXVAL 255\rTUPLTYPE RGB\rENDHDR\r");
  pam.resize(pam.size() + 18, 42);
  RoundTrip(pam, 6, "P7 CR-only");
  for (const char* resolution : {"-Y\t2 +X\t3\n", "-Y +2 +X +3\n", "-Y2+X3\n"}) {
    auto hdr = Bytes(std::string("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n") + resolution);
    // Width below eight uses the Radiance flat RGBE raster, four bytes/pixel.
    for (int pixel = 0; pixel < 6; ++pixel)
      hdr.insert(hdr.end(), {32, 64, 96, 128});
    RoundTrip(hdr, 6, resolution);
  }
  for (const char* resolution : {"-Y + 2 +X 3\n", "-Y ++2 +X 3\n",
                                 "-Y 2 +X -3\n", "-Y 0 +X 3\n",
                                 "-Y 2 +X +2147483648\n",
                                 "-Y 99999999999999999999 +X 3\n"}) {
    auto hdr = Bytes(std::string("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n") + resolution);
    Check(!PrepareEncodedImage(Base64(hdr)), "HDR: malformed or overflowing signed dimension rejected");
  }
  auto repeated_tuple = Bytes("P7\nWIDTH 3\nHEIGHT 2\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE RGB\nENDHDR\n");
  repeated_tuple.resize(repeated_tuple.size() + 18, 42);
  RoundTrip(repeated_tuple, 6, "P7 repeated TUPLTYPE");
  std::vector<uint8_t> bmp;
  Check(cv::imencode(".bmp", cv::Mat(2, 3, CV_8UC3, cv::Scalar(3, 4, 5)), bmp),
        "encode BMP boundary fixture");
  auto top_down = bmp;
  top_down[22] = 0xfe;
  top_down[23] = top_down[24] = top_down[25] = 0xff;
  RoundTrip(top_down, 6, "BMP top-down");
  auto bad = bmp;
  bad[22] = bad[23] = bad[24] = 0;
  bad[25] = 0x80;
  Check(!PrepareEncodedImage(Base64(bad)), "BMP INT_MIN height rejected without signed overflow");
  bad = bmp;
  bad[21] = 0x80;
  Check(!PrepareEncodedImage(Base64(bad)), "negative BMP width rejected");
  bad = bmp;
  bad[14] = bad[15] = bad[16] = bad[17] = 0xff;
  Check(!PrepareEncodedImage(Base64(bad)), "oversized DIB header rejected");
  for (size_t n = 0; n < 26; ++n)
    Check(!PrepareEncodedImage(Base64(std::span(bmp).first(n))), "truncated BMP header rejected");
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0xc0, 0xff, 0xff};
  Check(!PrepareEncodedImage(Base64(jpeg)), "JPEG segment beyond input rejected");
  auto truncated_payload = PrepareEncodedImage(Base64(Bytes("P6\n3 2\n255\n")));
  Check(truncated_payload && truncated_payload->pixels == 6,
        "header-only preflight does not decode missing raster");
  Check(!DecodeImageBytes(truncated_payload->bytes), "codec rejects missing raster after preflight");
}
void Orientation() {
  std::vector<uint8_t> jpeg;
  Check(cv::imencode(".jpg", cv::Mat(2, 3, CV_8UC3, cv::Scalar(1, 2, 3)), jpeg), "encode orientation fixture");
  // APP1 Exif little-endian TIFF: orientation 6 (90 degrees clockwise).
  const uint8_t exif[] = {0xff,0xe1,0,34,'E','x','i','f',0,0,'I','I',42,0,8,0,0,0,
                         1,0,0x12,1,3,0,1,0,0,0,6,0,0,0,0,0,0,0};
  jpeg.insert(jpeg.begin() + 2, std::begin(exif), std::end(exif));
  auto prepared = PrepareEncodedImage(Base64(jpeg));
  Check(prepared && prepared->pixels == 6, "EXIF orientation leaves prepared pixel count unchanged");
  auto normal = DecodeImageBytes(prepared->bytes);
  Check(normal && normal->rows == 3 && normal->cols == 2, "default decode still applies EXIF orientation");
  auto bounded = DecodeEncodedImage(Base64(jpeg), 6);
  Check(bounded && bounded->rows == 2 && bounded->cols == 3, "legacy bounded decode still ignores orientation");
}
}  // namespace
int main() {
  Formats();
  BadHeadersAndLimits();
  BinaryHeaderBoundaries();
  Orientation();
  std::cout << "image codec regressions passed\n";
}
