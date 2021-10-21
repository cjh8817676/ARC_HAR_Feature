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

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "hx_drv_tflm.h"

#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "model.h"
#include "test_samples.h"
#include "math.h"
#include "stdlib.h"
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
union float_c{
  float m_float;
  uint8_t m_bytes[sizeof(float)];
};
union int_c{
  float m_int;
  uint8_t m_bytes[sizeof(int)];
};
typedef struct
{
	uint8_t symbol;
	uint32_t int_part;
	uint32_t frac_part;
} accel_type;
union float_c myfloat;
union int_c myint;
// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 200 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize] = {0};
}  // namespace


volatile void delay_ms(unsigned int ms);


//--------------------------------------------------------------------//

void swap(float* a, float* b)
{
    float t = *a;
    *a = *b;
    *b = t;
}
/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
float partition (float arr[], int low, int high)
{
    float pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
 
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void quickSort(float *arr, int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
 
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
 
float find_median(float list[],int size = 100)
{
    quickSort(list,0,size-1);
    if (size % 2 != 0)
        return (double)list[size / 2];
 
    return (double)(list[(size - 1) / 2] + list[size / 2]) / 2.0;
}

int median_index(float* list, int l, int r)
{
    int n = r - l + 1;
    n = (n + 1) / 2 - 1;
    return n + l;
}

float Percentile(float sequence[] , float excelPercentile,int size)
{
    quickSort(sequence,0,size-1);
    int N = size;
    double n = (N - 1) * excelPercentile + 1;
    // Another method: double n = (N + 1) * excelPercentile;
    if (n == 1) return sequence[0];
    else if (n == N) return sequence[N - 1];
    else
    {
         int k = (int)n;
         double d = n - k;
         return sequence[k - 1] + d * (sequence[k] - sequence[k - 1]);
    }
}
//----------------------------特徵值-----------------------------------//
float find_std(float list[], int size = 100) //'x_std' 'y_std' 'z_std'
{
    float sum = 0.0, mean, SD = 0.0;
    int i;
    for (i = 0; i < size; ++i) {
        sum += list[i];
    }
    mean = sum / size;
   
    for (i = 0; i < size; ++i) {
        SD += pow(list[i] - mean, 2);
    }
    return sqrt(double(SD / size));
}

float find_min(float list[], int size = 100) //'x_min' 'y_min' 'z_min'
{
    float min;
    min = list[0];
    for (int i = 0;i<size;i++)
    {
        if (list[i] < min)
            min = list[i];
    }
    return min;
}

float find_max(float list[],int size = 100) //'x_max' 'y_max' 'z_max'
{
    float max;
    max = list[0];
    for (int i = 0;i<size;i++)
    {
        if (list[i] > max)
            max = list[i];
    }
    return max;
}

float find_aad(float list[], int size = 100) //avg absolute diff
{
    float mean,sum;
    float aad;
    for (int i = 0; i < size; ++i) {
        sum += list[i];
    }
    mean = sum / size;
    sum = 0;

    for (int i = 0; i < size; ++i) {
        sum += abs(list[i] - mean);
    }

    aad = sum / size;

    return aad;
}

float find_maxmin_diff(float list[], int size = 100)
{
    float max,min;
    max = find_max(list);
    min = find_min(list);

    return max - min;
}

float find_mad(float list[],int size = 100) // median abs dev
{
    float sum = 0,median;
    float temp[100];
    median = find_median(list,100);

    sum = 0;
    for (int i = 0; i < size; i++)
      temp[i] = abs(list[i] - median);

   return find_median(temp,100);
}

float find_IQR(float list[],int size = 100) //interquartile range // 百分等級!!!
{
    float Q1 = Percentile(list,0.25,size);
    float Q3 = Percentile(list,0.75,size);

    //std::cout << Q1 << " " << Q3 << std::endl;

    return (Q3 - Q1);
}

float find_neg_count(float list[],int size = 100)
{
    float neg_count;

    for (int i = 0 ; i < size ; i++)
    {
        if (list[i] < 0)
            neg_count++;
    }
    return  neg_count;
}

float find_pos_count(float list[],int size = 100)
{
    float pos_count;

    for (int i = 0 ; i < size ; i++)
    {
        if (list[i] > 0)
            pos_count++;
    }
    return  pos_count;
}

float find_energy(float list[],int size = 100)
{
    float energy;

    for (int i=0; i < size; i++)
    {
        energy += list[i] * list[i];
    }
    return energy / size;
}

float find_avg_result_accl(float **list,int size = 100)
{
    float avg_result_accl;
    for (int i = 0 ; i < size; i++)
    {
        avg_result_accl += pow( (pow(list[0][i],2) + pow(list[1][i],2) + pow(list[2][i],2)) , 0.5);
    }
    return avg_result_accl / size;
}

float find_sma(float **list, int size = 100) //signal magnitude area
{
    float sma;
    float sma_x,sma_y,sma_z;
    for (int i = 0 ; i < size; i++)
    {
        sma_x += abs(list[0][i])/100 ;
        sma_y += abs(list[1][i])/100 ;
        sma_z += abs(list[2][i])/100 ;
    }
    return sma_x + sma_y + sma_z;
}

float find_argmax(float list[], int size = 100)
{
    float index;
    float max = find_max(list);
    for (int i = 0 ;i < size ; i++)
    {
        if(list[i] == max)
        {
            index =  i;
            break;
        }
    }
    return index;
}


float feature_funcptr[30];  
//--------------------------------------------------------------------//

// The name of this function is important for Arduino compatibility.
void setup() {
  //hal_gpio_get(&hal_gpio_0, &gpio_level);
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  
  static tflite::MicroMutableOpResolver<3> micro_op_resolver;
  //micro_op_resolver.AddConv2D();
  //micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  //micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  //micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddRelu();
  //micro_op_resolver.AddSplit();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Obtain quantization parameters for result dequantization
}

// The name of this function is important for Arduino compatibility.
void loop_har(float x_data[3][600],uint8_t *signal_pass) {
  char string_buf[100];
	int32_t test_cnt = 0;
	int32_t correct_cnt = 0;
	float scale = input->params.scale;
	int32_t zero_point = input->params.zero_point;
  float x_data_temp[3][100];
  float *data_ptr[3];

  for (int i = 0 ;i< 600; i = i+6)
    {
       x_data_temp[0][(i/6)] = x_data[0][i];
       x_data_temp[1][(i/6)] = x_data[1][i];
       x_data_temp[2][(i/6)] = x_data[2][i];
    }
    data_ptr[0] = x_data_temp[0];
    data_ptr[1] = x_data_temp[1];
    data_ptr[2] = x_data_temp[2];
    
    for (int i = 0; i< 100;i++)
    {
        data_ptr[0][i] = (data_ptr[0][i] + 4) / 8;
        
        data_ptr[1][i] = (data_ptr[1][i] + 4) / 8;
       
        data_ptr[2][i] = (data_ptr[2][i] + 4) / 8;
    }


    feature_funcptr[0] = find_std(data_ptr[0]); //x_std
    feature_funcptr[1] = find_std(data_ptr[1]); //y_std
    feature_funcptr[2] = find_std(data_ptr[1]); //z_std
    feature_funcptr[3] = find_aad(data_ptr[0]); //x_aad
    feature_funcptr[4] = find_aad(data_ptr[1]); //y_aad
    feature_funcptr[5] = find_aad(data_ptr[2]); //z_aad
    feature_funcptr[6] = find_min(data_ptr[0]); //x_min
    feature_funcptr[7] = find_min(data_ptr[1]); //y_min
    feature_funcptr[8] = find_min(data_ptr[2]); //z_min
    feature_funcptr[9] = find_max(data_ptr[0]); //x_max
    feature_funcptr[10] =find_max(data_ptr[1]); //y_max
    feature_funcptr[11] =find_maxmin_diff(data_ptr[0]); //x_maxmin_diff
    feature_funcptr[12] =find_maxmin_diff(data_ptr[1]); //y_maxmin_diff
    feature_funcptr[13] =find_maxmin_diff(data_ptr[2]); //z_maxmin_diff
    feature_funcptr[14] =find_mad(data_ptr[0]); //x_mad
    feature_funcptr[15] =find_mad(data_ptr[1]); //y_mad
    feature_funcptr[16] =find_mad(data_ptr[2]); //z_mad
    feature_funcptr[17] =find_IQR(data_ptr[0]); //x_IQR
    feature_funcptr[18] =find_IQR(data_ptr[1]); //y_IQR
    feature_funcptr[19] =find_IQR(data_ptr[2]); //z_IQR
    //   feature_funcptr[20] =find_neg_count(data_ptr[1]); //y_neg_count
    //   feature_funcptr[21] =find_neg_count(data_ptr[2]); //z_neg_count
    //   feature_funcptr[22] =find_pos_count(data_ptr[1]); //y_pos_count
    //   feature_funcptr[23] =find_pos_count(data_ptr[2]); //z_pos_count
    //   feature_funcptr[24] =find_energy(data_ptr[0]); //x_energy
    //   feature_funcptr[25] =find_energy(data_ptr[1]); //y_energy
    //   feature_funcptr[26] =find_energy(data_ptr[2]); //z_energy
    //   feature_funcptr[27] =find_avg_result_accl(data_ptr); //avg_result_accl
    //   feature_funcptr[28] =find_sma(data_ptr); // sma
    //   feature_funcptr[29] =find_argmax(data_ptr[1]); //y_argmax

    //for (int j = 0; j < 50; j++)
    //{
		//TF_LITE_REPORT_ERROR(error_reporter, "Test sample[%d] Start Reading and round values to either -128 or 127\n", j);
		// Perform image thinning (round values to either -128 or 127)
		// Write image to input data
		for (int i = 0; i < 20; i=i+1) {  //  (int i = 0; i < 50; i=i++)(三軸輸入)        (int i = 0; i < 150; i=i+1)(temple.cc輸入)
            input->data.f[i] = feature_funcptr[i];// x
			//input->data.f[3*i] = (x_data_temp[0][i] + 4 )/ 8;// x
            //input->data.f[3*i+1] = (x_data_temp[1][i] + 4)/ 8;// y
            //input->data.f[3*i+2] = (x_data_temp[2][i] + 4)/ 8 ;// z
            //TF_LITE_REPORT_ERROR(error_reporter, "%f | %f | %f \n ", test_samples[j].image[i], test_samples[j].image[i+1], test_samples[j].image[i+2]);
            //input->data.f[i] = test_samples[j].image[i];
		}

		//TF_LITE_REPORT_ERROR(error_reporter, "Test sample[%d] Start Invoking\n", j);
		// Run the model on this input and make sure it succeeds.   !!!!!!!!!!!!!!!!
		if (kTfLiteOk != interpreter->Invoke()) {
			TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
		}

		//TF_LITE_REPORT_ERROR(error_reporter, "Test sample[%d] Start Finding Max Value\n", j);
		// Get max result from output array and calculate confidence
		float* results_ptr = output->data.f;
		//TF_LITE_REPORT_ERROR(error_reporter, "1");
		int result = std::distance(results_ptr, std::max_element(results_ptr, results_ptr + 6));
		//TF_LITE_REPORT_ERROR(error_reporter, "2");
		float confidence = ((results_ptr[result] - zero_point)*scale + 1) / 2;
		//TF_LITE_REPORT_ERROR(error_reporter, "3");
		//const char *status = result == test_samples[j].label ? "SUCCESS" : "FAIL";
		//TF_LITE_REPORT_ERROR(error_reporter, "4");

       *signal_pass = result;
		//if(result == test_samples[j].label)
		//	correct_cnt ++;
		//test_cnt ++;
		//TF_LITE_REPORT_ERROR(error_reporter, "5");
    TF_LITE_REPORT_ERROR(error_reporter, 
      
      "Predicted %s  \n",
      
      kCategoryLabels[result]);   
		//TF_LITE_REPORT_ERROR(error_reporter, "6");

		//TF_LITE_REPORT_ERROR(error_reporter, "Correct Rate = %d / %d\n\n", correct_cnt, test_cnt);
		//TF_LITE_REPORT_ERROR(error_reporter, "7");
  //}
}



