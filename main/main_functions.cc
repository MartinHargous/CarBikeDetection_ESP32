/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "esp_psram.h"
#include "esp_task_wdt.h"
// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 375 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;
//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

void print_memory_statistics(const char* tag) {
  printf("%s\n", tag);
  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Total PSRAM size: %d\n", heap_caps_get_total_size(MALLOC_CAP_SPIRAM));
  printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

void setup() {
  // Print memory statistics before allocation
  print_memory_statistics("Before allocation");

  // Attempt to allocate memory in PSRAM first
  tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes in PSRAM.\n", kTensorArenaSize);
    // Print memory statistics after failed PSRAM allocation attempt
    print_memory_statistics("After PSRAM allocation attempt");

    // Attempt to allocate memory in internal memory
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == NULL) {
      printf("Couldn't allocate memory of %d bytes in internal memory either.\n", kTensorArenaSize);
      // Print memory statistics after failed internal memory allocation attempt
      print_memory_statistics("After internal memory allocation attempt");
      return;
    } else {
      printf("Successfully allocated %d bytes in internal memory.\n", kTensorArenaSize);
      // Print memory statistics after successful internal memory allocation
      print_memory_statistics("After successful internal memory allocation");
    }
  } else {
    printf("Successfully allocated %d bytes in PSRAM.\n", kTensorArenaSize);
    // Print memory statistics after successful PSRAM allocation
    print_memory_statistics("After successful PSRAM allocation");
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) return;
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) return;
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) return;
  if (micro_op_resolver.AddReshape() != kTfLiteOk) return;
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return;
  if (micro_op_resolver.AddDequantize() != kTfLiteOk) return;
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.

  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }


  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t car_score = output->data.uint8[kCarIndex];
  int8_t bike_score = output->data.uint8[kBikeIndex];

  float car_score_f =
      (car_score - output->params.zero_point) * output->params.scale;
  float bike_score_f =
      (bike_score - output->params.zero_point) * output->params.scale;

  // Respond to detection
  RespondToDetection(car_score_f, bike_score_f);

  vTaskDelay(1); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
  extern long long reshape_total_time;
  extern long long quantize_total_time;
  extern long long dequantize_total_time;
#endif

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
    //printf("%d, ", input->data.int8[i]);
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  printf("Softmax time = %lldmicroSeg\n", softmax_total_time);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);
  printf("reshape time = %lldmicroSeg\n", reshape_total_time);
  printf("quantize time = %lldmicroSeg\n", quantize_total_time);
  printf("dequantize time = %lldmicroSeg\n", quantize_total_time);
  //int cpu_hz = 240000000;
  //float total_cycles = (total_time/1000) * cpu_hz;
  //float dc_total_cycles = (dc_total_time/1000) * cpu_hz;
  //float conv_total_cycles = (conv_total_time/1000) * cpu_hz;
  //float fc_total_cycles = (fc_total_time/1000) * cpu_hz;
  //float pooling_total_cycles = (pooling_total_time/1000) * cpu_hz;
  //float add_total_cycles = (add_total_time/1000) * cpu_hz;
  //float mul_total_cycles = (mul_total_time/1000) * cpu_hz;
//
//
  //float cpi_avg_total = total_cycles/3368;
  //float cpi_fc = fc_total_cycles/222;
  //float cpi_conv = conv_total_cycles/448;
  //float cpi_pool = pooling_total_cycles/720;
//
  //printf("CPI TOTAL AVG = %f\n", cpi_avg_total);
  //printf("CPI SOFTMAX AVG = 27360\n");
  //printf("CPI FC AVG = %f\n", cpi_fc);
  //printf("CPI CONV AVG = %f\n", cpi_conv);
  //printf("CPI MAXPOOL AVG = %f\n", cpi_pool);
  //printf("CPI RESHAPE AVG = 10080\n");
  //printf("CPI QUANT AVG = 115200\n");
  //printf("CPI DEQUANT AVG = 115200\n");

  //float conv2d_ops = (conv_total_time/1000)/448;
  //float fc_ops = (fc_total_time/1000)/222;
  //float pool_ops = (pooling_total_time/1000)/720;
  //float reshape_ops = (reshape_total_time/1000)288;
  //float softmax_ops = (softmax_total_time/1000)/891;
  //float quant_ops = (quantize_total_time/1000)/349;
  //float dequant_ops = (dequantize_total_time/1000)/450;

  /* Reset times */
  total_time = 0;
  softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
  reshape_total_time = 0;
  quantize_total_time = 0;
  dequantize_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);
  printf("Input type: %s\n", TfLiteTypeGetName(input->type));
  printf("Output type: %s\n", TfLiteTypeGetName(output->type));

  // Process the inference results.
  float car_score = output->data.uint8[kCarIndex];
  float bike_score = output->data.uint8[kBikeIndex];
  float car_score_f =
      (car_score - output->params.zero_point) * output->params.scale;
  float bike_score_f =
      (bike_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(car_score_f, bike_score_f);

}