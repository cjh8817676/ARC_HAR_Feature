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
#include "hx_drv_tflm.h"
#include "stdio.h"
#include "math.h"

//#include "synopsys_wei_gpio.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
// int signal_pass=0;
// volatile void delay_ms(unsigned int ms);
// volatile void delay_us(unsigned int ms);

char string_buf[100] = "test\n";

#define accel_scale 1000

typedef struct
{
	uint8_t symbol;
	uint32_t int_part;
	uint32_t frac_part;
} accel_type;
volatile void delay_ms(unsigned int ms)
{
  volatile unsigned int delay_i = 0;
  volatile unsigned int delay_j = 0;
  volatile const unsigned int delay_cnt = 20000;

  for(delay_i = 0; delay_i < ms; delay_i ++)
    for(delay_j = 0; delay_j < delay_cnt; delay_j ++);
} 
volatile void delay_us(unsigned int us)
{
  uint32_t start = 0;
  uint32_t end = 0;
  uint32_t diff = 0;
  hx_drv_tick_get(&start);	

  do{
	hx_drv_tick_get(&end);	
	if(end < start){
		diff = 0XFFFFFFFF - start + end;
	}else{
		diff = (end - start);
	}
  }while( diff < (us * 400) );
} 

void GPIO_UART_TX(uint8_t *data, uint32_t len , hx_drv_gpio_config_t *Tx) //gpio0 為板子的Tx
{
	for ( int i = 0; i < len; i++ )
    {
        uint8_t c = data[i];
        
        // pull down as start bit
		Tx -> gpio_data = 0;
        hx_drv_gpio_set(Tx);
        delay_us(26);  // baudrate默認 38400
        for ( int j = 0; j < 8; j++ )
        {
            if ( c & 0x01 )
            {
				Tx->gpio_data = 1;
                hx_drv_gpio_set(Tx);  //GPIO_PIN_SET = 1
            }
            else
            {
				Tx->gpio_data = 0;
                hx_drv_gpio_set(Tx);
            }
            delay_us(26);
            c >>= 1;
        }
        // pull high as stop bit
		Tx->gpio_data = 1;
        hx_drv_gpio_set(Tx);
        delay_us(26);
    }
}
void GPIO_UART_RX(uint8_t *data, uint16_t *len, hx_drv_gpio_config_t * Rx) //gpio1 為板子的Rx
{

	int count = 0; 
	while(1)
    {
        uint8_t b[8] = { 0 };
        uint32_t c = 0;
        // wait for start bit
		do{
			hx_drv_gpio_get(Rx);
		}while ( Rx->gpio_data == 1 );  // wait while gpio read high
        delay_us(26/2);
		//delay_us(26);
        for ( int j = 0; j < 8; j++ )
        {
            delay_us(26);
			hx_drv_gpio_get(Rx);
            b[j] = ( Rx->gpio_data) ? 1 : 0;
        }
        
        for ( int j = 0; j < 8; j++ )
        {
            c |= (b[j] << j);
        }
        data[count & 0x3fff] = c;

		count++;
		if(c == 0x0d){   // 0x0d'\r'
				
			count |= 0x8000;
		}else if((c == 0x0a) && (count & 0x8000) ){ //0x0a '\n'
			count |= 0x4000;
		}
		if( count & 0x4000){
			*len = (count & 0x3fff) - 2;
			data[*len] = '\0';
			break;
		}
        // wait for stop bit
		do{
			hx_drv_gpio_get(Rx);
		}
        while ( Rx->gpio_data == 0 );  //wait until gpio read high as stop bit
    }
}

static uint16_t sample_data = 600;
static uint16_t sample_data_nim_of_bytes = 17;

int main(int argc, char* argv[])
{
	float teeeerewr = 0.25;
	float teeeerewr2;
	teeeerewr2 = sqrt(teeeerewr);
	float x_data[3][600];
	uint8_t signal_pass;
	uint8_t label;
	int32_t int_buf;
	accel_type accel_x, accel_y, accel_z;
	uint32_t msec_cnt = 0;
	uint32_t sec_cnt = 0;
	uint16_t counter = 0;
	int32_t flagx,flagy,flagz;
	uint16_t str_length;
	hx_drv_gpio_config_t hal_gpio_ZERO;
	hx_drv_gpio_config_t hal_gpio_ONE;

	hx_drv_uart_initial(UART_BR_115200);

	//It will initial accelerometer with sampling rate 119 Hz, bandwidth 50 Hz, scale selection 4g at continuous mode.
	//Accelerometer operates in FIFO mode. 
	//FIFO size is 32

	hal_gpio_ZERO.gpio_pin = HX_DRV_PGPIO_0;  // 決定pin角要用數字
    hal_gpio_ZERO.gpio_direction = HX_DRV_GPIO_OUTPUT;  //HX_DRV_GPIO_OUTPUT(3)
    hal_gpio_ZERO.gpio_data = 1;
	
	if(hx_drv_gpio_initial(&hal_gpio_ZERO) != HX_DRV_LIB_PASS)
    	hx_drv_uart_print("Accel 0 fail");
  	else
    	hx_drv_uart_print("Accel 0 success");

	hal_gpio_ONE.gpio_pin = HX_DRV_PGPIO_1;  // 決定pin角要用數字
    hal_gpio_ONE.gpio_direction = HX_DRV_GPIO_INPUT;  //HX_DRV_GPIO_OUTPUT(3)
    hal_gpio_ONE.gpio_data = 0;

	
	if(hx_drv_gpio_initial(&hal_gpio_ONE) != HX_DRV_LIB_PASS)
    	hx_drv_uart_print("Accel 1 fail");
  	else
    	hx_drv_uart_print("Accel 1 success");


	if (hx_drv_accelerometer_initial() != HX_DRV_LIB_PASS)
		hx_drv_uart_print("Accelerometer Initialize Fail\n");
	else
		hx_drv_uart_print("Accelerometer Initialize Success\n");
	setup();
  
	uint8_t in;
	uint8_t temp_a[sample_data_nim_of_bytes];
	delay_ms(10);
	while (1) 
	{
		uint32_t available_count = 0;
		float x, y, z;
		//available_count = hx_drv_accelerometer_available_count();				
		//hx_drv_uart_print("Accel get FIFO: %d\n", available_count);
		/*for (int i = 0; i < available_count; i++) 
		{
			hx_drv_accelerometer_receive(&x, &y, &z);
		}*/
		for(int k=0;k<sample_data_nim_of_bytes;k++){
			temp_a[k] = 0;   
		}
		hx_drv_accelerometer_receive(&x, &y, &z);

		x_data[0][counter] = x;
		x_data[1][counter] = y;
		x_data[2][counter] = z;

		int_buf = x_data[0][counter] * accel_scale; //scale value
		if(int_buf < 0)
		{
			int_buf = int_buf * -1;
			accel_x.symbol = '-';
			flagx = -1;
		}
		else 
		{
			accel_x.symbol = '+';
			flagx = 1;
		}
		accel_x.int_part = int_buf / accel_scale;
		accel_x.frac_part = int_buf % accel_scale;


		int_buf = x_data[1][counter] * accel_scale; //scale value
		if(int_buf < 0)
		{
			int_buf = int_buf * -1;
			accel_y.symbol = '-';
			flagy = -1;
		}
		else 
		{
			accel_y.symbol = '+';
			flagy = 1;
		}
		accel_y.int_part = int_buf / accel_scale;
		accel_y.frac_part = int_buf % accel_scale;


		int_buf = x_data[2][counter] * accel_scale; //scale value
		if(int_buf < 0)
		{
			int_buf = int_buf * -1;
			accel_z.symbol = '-';
			flagz = -1;
		}
		else 
		{
			accel_z.symbol = '+';
			flagz = 1;
		}
		accel_z.int_part = int_buf / accel_scale;
		accel_z.frac_part = int_buf % accel_scale;

		temp_a[0] = 0 ; 
		temp_a[1] = counter >> 8;     //高8位
		temp_a[2] = counter & 0xff ;  //低8位
		temp_a[3] = accel_x.symbol == '+';
		temp_a[4] = accel_x.int_part;
		temp_a[5] = accel_x.frac_part & 0XFF;
		temp_a[6] = accel_x.frac_part >> 8;
		temp_a[7] = accel_y.symbol == '+';
		temp_a[8] = accel_y.int_part ;
		temp_a[9] = accel_y.frac_part & 0XFF;
		temp_a[10] = accel_y.frac_part >> 8;
		temp_a[11] = accel_z.symbol == '+';
		temp_a[12] = accel_z.int_part; 
		temp_a[13] = accel_z.frac_part & 0XFF; 
		temp_a[14] = accel_z.frac_part >> 8; 
		temp_a[15] = 0xff; 
		temp_a[16] = 0xff;

		/*if (counter == 0)
		{
			temp_a[1] = 0x00;
			temp_a[2] = 0x00;
		}*/


		uint8_t sum = 0;
		for(int k=1;k<15;k++){
			sum += temp_a[k];
		}
		temp_a[0] = sum;
		// for (int j = 0;j<sample_data_nim_of_bytes;j++)
		// {
		// 	a[counter][j] = temp_a[j] ;
		// 	delay_us(1);  // 放太快會錯
		// }

		sprintf(string_buf, "%d | %c%d.%3d | %c%d.%3d | %c%d.%3d \n", 
				counter ,accel_x.symbol, accel_x.int_part, accel_x.frac_part, 
				accel_y.symbol, accel_y.int_part, accel_y.frac_part, 
				accel_z.symbol, accel_z.int_part, accel_z.frac_part);
		hx_drv_uart_print(string_buf);
		
		
		//hx_drv_uart_print("\n\n");
		
		msec_cnt = msec_cnt + 125;
		sec_cnt = sec_cnt + (msec_cnt / 1000);
		msec_cnt = msec_cnt % 1000;
		
		
		GPIO_UART_TX(temp_a,sample_data_nim_of_bytes,&hal_gpio_ZERO);
		// do {
		// 	GPIO_UART_RX(&in,&str_length,&hal_gpio_ONE);
		// }while( in != NULL );
		counter++;
		if (counter >= 600)
		{
			hx_drv_led_on(HX_DRV_LED_RED);
			// hal_gpio_ZERO.gpio_data = 1;
        	// hx_drv_gpio_set(&hal_gpio_ZERO);
			counter = 0;
			// // for (uint16_t k=0;k<sample_data;k++){
			// // 	/*
			// // 	sprintf(string_buf, "%d | %d | %d \n", 
			// // 		a[k][1] << 8 , a[k][2], k);
			// // 		hx_drv_uart_print(string_buf);*/
			// // 	uint8_t sum = 0;
			// // 	a[k][1] = k >> 8;	
			// // 	a[k][2] = k & 0xff;	
			// // 	a[k][0] = sum;	
				
			// // 	GPIO_UART_TX(a[k],sample_data_nim_of_bytes,&hal_gpio_ZERO);
			// // 	//GPIO_UART_RX(&in,&str_length,&hal_gpio_ONE);
			// // 	/*sprintf(string_buf, " %d | sum:%d | send:%d \n" ,k, sum , in);
			// // 	hx_drv_uart_print(string_buf);*/
			// // }
			loop_har(x_data,&signal_pass);
			delay_ms(1500);
			label = signal_pass;
			GPIO_UART_TX(&label,1,&hal_gpio_ZERO);

			// // sprintf(string_buf, "here : %d \n", label);
			// // hx_drv_uart_print(string_buf);
			// // hx_drv_uart_print("\n");
			hx_drv_led_off(HX_DRV_LED_RED);
			hal_gpio_ZERO.gpio_data = 1;
        	hx_drv_gpio_set(&hal_gpio_ZERO);

			for (int i = 0 ;i< 600;i++)
			{	
				x_data[0][i] = 0;
				x_data[1][i] = 0;
				x_data[2][i] = 0;
			}
			/*
			while(1){
				hx_drv_led_on(HX_DRV_LED_RED);
				delay_ms(100);
				hx_drv_led_off(HX_DRV_LED_RED);
				delay_ms(100);
			}*/
			// delay_ms(1000);
		}
		
	}




	// label = 0x84;
	// delay_us(260);
	// GPIO_UART_TX(&label,1,&hal_gpio_ZERO);
	// delay_us(260);


	// hx_drv_uart_print("stop here");
	// for (int i = 0 ;i< 300;i++)
	// {	
	// 	x_data[0][i] = 0;
	// 	x_data[1][i] = 0;
	// 	x_data[2][i] = 0;
	// }

	// while(1){
	// 	hx_drv_led_on(HX_DRV_LED_RED);
	// 	delay_ms(100);
	// 	hx_drv_led_off(HX_DRV_LED_RED);
	// 	delay_ms(100);
	// }
}