/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Unit tests for amdsmi_uuid_gen(); no GPU required. The generator is a pure
// function of (serial, did, function index), so its canonical 8-4-4-4-12 output
// and its buffer bounds are fully verifiable in isolation. These tests pin the
// formatting contract that the bounded-snprintf rewrite must preserve and guard
// against a regression back to an unbounded writer overrunning the UUID buffer.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_uuid.h"

namespace {

bool IsLowerHex(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

// A canonical UUID is exactly 36 chars: 8-4-4-4-12 lowercase hex with hyphens
// at indices 8, 13, 18, 23 and a NUL at 36.
void ExpectCanonicalShape(const char* uuid) {
  ASSERT_EQ(std::strlen(uuid), 36u);
  for (int i = 0; i < 36; ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      EXPECT_EQ(uuid[i], '-') << "expected hyphen at index " << i;
    } else {
      EXPECT_TRUE(IsLowerHex(uuid[i])) << "expected lowercase hex at index " << i;
    }
  }
}

}  // namespace

TEST(UuidGen, FormatsCanonicalLayoutForKnownInputs) {
  char uuid[AMDSMI_GPU_UUID_SIZE] = {};

  ASSERT_EQ(amdsmi_uuid_gen(uuid, /*serial=*/0xDEADBEEFull, /*did=*/0x1234, /*idx=*/0x56),
            AMDSMI_STATUS_SUCCESS);

  ExpectCanonicalShape(uuid);
  // version nibble is fixed at 1 (first char of the third group).
  EXPECT_EQ(uuid[14], '1');
  // The ASIC serial's low 32 bits land verbatim in the trailing 8 hex digits.
  EXPECT_STREQ(uuid + 24, "0000deadbeef");
}

TEST(UuidGen, StaysWithinUuidBufferOnMaxInputs) {
  // Oversize the buffer and fence it with a canary so any write past the
  // AMDSMI_GPU_UUID_SIZE-bounded region is observable.
  constexpr char kCanary = 0x7e;
  char uuid[AMDSMI_GPU_UUID_SIZE + 8];
  std::memset(uuid, kCanary, sizeof(uuid));

  ASSERT_EQ(amdsmi_uuid_gen(uuid, /*serial=*/0xFFFFFFFFFFFFFFFFull, /*did=*/0xFFFF, /*idx=*/0xFF),
            AMDSMI_STATUS_SUCCESS);

  ExpectCanonicalShape(uuid);
  EXPECT_EQ(uuid[36], '\0');
  // 36 chars + NUL must fit within AMDSMI_GPU_UUID_SIZE; bytes beyond it untouched.
  for (size_t i = AMDSMI_GPU_UUID_SIZE; i < sizeof(uuid); ++i) {
    EXPECT_EQ(uuid[i], kCanary) << "buffer overrun at index " << i;
  }
}
