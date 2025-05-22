#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

//stb image headers
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

//cuda kernel to detect edges within the given imagedata
__global__ void edge_detection_on_gpu( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //sobel edge detection
    const int convolution_matrix_1[3][3] = {
        {  1,  0, -1 },
        {  2,  0, -2 },
        {  1,  0, -1 }
    };

    const int convolution_matrix_2[3][3] = {
        {  1,  2,  1 },
        {  0,  0,  0 },
        { -1, -2, -1 }
    };

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        if ( ( pixelX > 2 ) && ( pixelY > 2 ) )
        {
            int final_sum_1_r = 0;
            int final_sum_1_g = 0;
            int final_sum_1_b = 0;
            int final_sum_2_r = 0;
            int final_sum_2_g = 0;
            int final_sum_2_b = 0;

            //do the convolution
            for ( int y = pixelY; y > ( pixelY - 3 ); y-- )
            {
                for ( int x = pixelX; x > ( pixelX - 3 ); x-- )
                {
                    int convolution_value_1 = ( convolution_matrix_1[ -( x - pixelX ) ][ -( y - pixelY ) ] );
                    int convolution_value_2 = ( convolution_matrix_2[ -( x - pixelX ) ][ -( y - pixelY ) ] );
                    final_sum_1_r += ( ( image_data[ ( x + y * image_width ) * component_count + 0 ] ) * convolution_value_1 );
                    final_sum_1_g += ( ( image_data[ ( x + y * image_width ) * component_count + 1 ] ) * convolution_value_1 );
                    final_sum_1_b += ( ( image_data[ ( x + y * image_width ) * component_count + 2 ] ) * convolution_value_1 );
                    final_sum_2_r += ( ( image_data[ ( x + y * image_width ) * component_count + 0 ] ) * convolution_value_2 );
                    final_sum_2_g += ( ( image_data[ ( x + y * image_width ) * component_count + 1 ] ) * convolution_value_2 );
                    final_sum_2_b += ( ( image_data[ ( x + y * image_width ) * component_count + 2 ] ) * convolution_value_2 );
                }
            }

            //combine the two convolutions
            uint8_t final_sum_r = ( uint8_t )( ( int )sqrtf( final_sum_1_r * final_sum_1_r + final_sum_2_r * final_sum_2_r ) & 0xFF );
            uint8_t final_sum_g = ( uint8_t )( ( int )sqrtf( final_sum_1_g * final_sum_1_g + final_sum_2_g * final_sum_2_g ) & 0xFF );
            uint8_t final_sum_b = ( uint8_t )( ( int )sqrtf( final_sum_1_b * final_sum_1_b + final_sum_2_b * final_sum_2_b ) & 0xFF );

            //write data to the correct memory address for the new image ( -3 pixels )
            output_data[ ( ( pixelX - 3 ) + ( pixelY - 3 ) * ( image_width - 3 ) ) * component_count + 0 ] = final_sum_r;
            output_data[ ( ( pixelX - 3 ) + ( pixelY - 3 ) * ( image_width - 3 ) ) * component_count + 1 ] = final_sum_g;
            output_data[ ( ( pixelX - 3 ) + ( pixelY - 3 ) * ( image_width - 3 ) ) * component_count + 2 ] = final_sum_b;
            output_data[ ( ( pixelX - 3 ) + ( pixelY - 3 ) * ( image_width - 3 ) ) * component_count + 3 ] = 255;
        }
    }
}

//cuda kernel to convert image to grayscale
__global__ void gray_scale_on_gpu( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        //grayscale formula
        uint8_t pixel_value = ( uint8_t )( image_data[ ( pixelX + pixelY * image_width ) * component_count + 0 ] * 0.2126f + image_data[ ( pixelX + pixelY * image_width ) * component_count + 1 ] * 0.7152f + image_data[ ( pixelX + pixelY * image_width ) * component_count + 2 ] * 0.0722f);
        
        //write data to the correct memory address for the new image
        output_data[ ( pixelX + pixelY * image_width ) * component_count + 0 ] = pixel_value;
        output_data[ ( pixelX + pixelY * image_width ) * component_count + 1 ] = pixel_value;
        output_data[ ( pixelX + pixelY * image_width ) * component_count + 2 ] = pixel_value;
        output_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ] = image_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ];
    }
}

//cuda kernel to do average pooling on image data
__global__ void average_pooling_on_gpu( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        //stride to a 2x2 grid
        if ( ( ( ( pixelX - 1 ) % 2 == 0 ) && ( ( pixelY - 1 ) % 2 == 0 ) ) )
        {
            int final_sum_r = 0;
            int final_sum_g = 0;
            int final_sum_b = 0;

            //get the sum of all values in a 2x2 square
            for ( int y = pixelY; y > ( pixelY - 2 ); y-- )
            {
                for ( int x = pixelX; x > ( pixelX - 2 ); x-- )
                {
                    final_sum_r += image_data[ ( x + y * image_width ) * component_count + 0 ];
                    final_sum_g += image_data[ ( x + y * image_width ) * component_count + 1 ];
                    final_sum_b += image_data[ ( x + y * image_width ) * component_count + 2 ];
                }
            }

            //write data to the correct memory address for the new image ( /2 )
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 0 ] = final_sum_r / 4; // sum devided by 4 for average formula
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 1 ] = final_sum_g / 4;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 2 ] = final_sum_b / 4;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 3 ] = image_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ];
        }
    }
}

//cuda kernel for doing max pooling on image data
__global__ void max_pooling_on_gpu( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        //stride to a 2x2 grid
        if ( ( ( ( pixelX - 1 ) % 2 == 0 ) && ( ( pixelY - 1 ) % 2 == 0 ) ) )
        {
            int final_max_r = 0;
            int final_max_g = 0;
            int final_max_b = 0;

            //find max value in a 2x2 grid
            for ( int y = pixelY; y > ( pixelY - 2 ); y-- )
            {
                for ( int x = pixelX; x > ( pixelX - 2 ); x-- )
                {
                    final_max_r = final_max_r > image_data[ ( x + y * image_width ) * component_count + 0 ] ? final_max_r : image_data[ ( x + y * image_width ) * component_count + 0 ];
                    final_max_g = final_max_g > image_data[ ( x + y * image_width ) * component_count + 1 ] ? final_max_g : image_data[ ( x + y * image_width ) * component_count + 1 ];
                    final_max_b = final_max_b > image_data[ ( x + y * image_width ) * component_count + 2 ] ? final_max_b : image_data[ ( x + y * image_width ) * component_count + 2 ];
                }
            }

            //write data to the correct memory address for the new image ( /2 )
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 0 ] = final_max_r;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 1 ] = final_max_g;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 2 ] = final_max_b;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 3 ] = image_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ];
        }
    }
}

//cuda kernel for doing minimum pooling on image data
__global__ void min_pooling_on_gpu( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        //stride to a 2x2 grid
        if ( ( ( ( pixelX - 1 ) % 2 == 0 ) && ( ( pixelY - 1 ) % 2 == 0 ) ) )
        {
            int final_min_r = 255;
            int final_min_g = 255;
            int final_min_b = 255;

            //find the lowest value starting from 255 ( highest )
            for ( int y = pixelY; y > ( pixelY - 2 ); y-- )
            {
                for ( int x = pixelX; x > ( pixelX - 2 ); x-- )
                {
                    final_min_r = final_min_r < image_data[ ( x + y * image_width ) * component_count + 0 ] ? final_min_r : image_data[ ( x + y * image_width ) * component_count + 0 ];
                    final_min_g = final_min_g < image_data[ ( x + y * image_width ) * component_count + 1 ] ? final_min_g : image_data[ ( x + y * image_width ) * component_count + 1 ];
                    final_min_b = final_min_b < image_data[ ( x + y * image_width ) * component_count + 2 ] ? final_min_b : image_data[ ( x + y * image_width ) * component_count + 2 ];
                }
            }

            //write data to the correct memory address for the new image ( /2 )
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 0 ] = final_min_r;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 1 ] = final_min_g;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 2 ] = final_min_b;
            output_data[ ( ( pixelX / 2 ) + ( pixelY / 2 ) * ( image_width / 2 ) ) * component_count + 3 ] = image_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ];
        }
    }
}

//kernel for copying data from the output image data to the input image data
__global__ void copy_output_to_input( uint8_t *image_data, uint8_t *output_data, int image_width, int image_height, int component_count )
{
    int pixelX = threadIdx.x + blockIdx.x * 32;
    int pixelY = threadIdx.y + blockIdx.y * 32;

    //only execute if the current pixel x and y are within the boundery of the image
    if ( ( component_count == 4 ) && ( ( pixelX < image_width ) && ( pixelY < image_height ) ) )
    {
        image_data[ ( pixelX + pixelY * image_width ) * component_count + 0 ] = output_data[ ( pixelX + pixelY * image_width ) * component_count + 0 ];
        image_data[ ( pixelX + pixelY * image_width ) * component_count + 1 ] = output_data[ ( pixelX + pixelY * image_width ) * component_count + 1 ];
        image_data[ ( pixelX + pixelY * image_width ) * component_count + 2 ] = output_data[ ( pixelX + pixelY * image_width ) * component_count + 2 ];
        image_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ] = output_data[ ( pixelX + pixelY * image_width ) * component_count + 3 ];
    }
}

class png_image_data
{
    private:
        const int required_component_count = 4; //r g b a

        dim3 *blockSize;
        dim3 *gridSize;

        uint8_t *all_pixel_values;
        uint8_t *output_pixel_values;
        uint8_t *ptr_image_data_on_gpu;
        uint8_t *ptr_output_data_on_gpu;

        int image_width;
        int image_height;
        int used_component_count;
        int element_count;

        enum png_error_states
        {
            PNG_ERROR_STATE_NONE,
            PNG_ERROR_STATE_CANNOT_OPEN_PNG,
            PNG_ERROR_STATE_COMPONENT_COUNT
        } png_error_state;

    public:
        enum png_kernel_option_t
        {
            PNG_EDGE_DETECTION,
            PNG_GRAY_SCALE,
            PNG_AVERAGE_POOLING,
            PNG_MAX_POOLING,
            PNG_MIN_POOLING
        };
        std::vector< enum png_kernel_option_t > png_kernel_options;
        
        png_image_data( std::string filename )
        {
            std::cout << "loading .png file " << filename << "\n\r";
            this->all_pixel_values = stbi_load( filename.c_str(), &( this->image_width ), &( this->image_height ), &( this->used_component_count ), this->required_component_count );
            
            //make sure the image got read correctly
            if ( !( this->all_pixel_values ) )
            {
                std::cout << "error: failed to open file\r\n";
                this->png_error_state = PNG_ERROR_STATE_CANNOT_OPEN_PNG;
            }
            else if ( this->used_component_count != this->required_component_count )
            {
                std::cout << "error: wrong image format ( component count = " << this->used_component_count << " )\r\n";
                this->png_error_state = PNG_ERROR_STATE_COMPONENT_COUNT;
            }
            else
            {
                //if image got read correclty parse the newly given image value's into the class
                this->element_count = this->image_width * this->image_height * this->used_component_count;
                std::cout << "data read: " << std::to_string( this->element_count ) << " elements\r\n";
                
                this->output_pixel_values = new uint8_t[ this->element_count ];
                for ( int i = 0; i < this->element_count; ++i )
                {
                    this->output_pixel_values[i] = this->all_pixel_values[i];
                }

                //cuda kernel value's
                ptr_image_data_on_gpu = nullptr;
                ptr_output_data_on_gpu = nullptr;

                blockSize = new dim3( 32, 32 );
                gridSize = new dim3( ( image_width + 32 - ( image_width % 32 ) ) / this->blockSize->x, ( image_height + 32 - ( image_height % 32 ) ) / this->blockSize->y );

                this->png_error_state = PNG_ERROR_STATE_NONE;
            }
        }

        ~png_image_data()
        {
            stbi_image_free( this->all_pixel_values );
            delete this->all_pixel_values;
        }

        int get_last_error()
        {
            switch ( this->png_error_state )
            {
                case PNG_ERROR_STATE_CANNOT_OPEN_PNG:
                    return -11;
                case PNG_ERROR_STATE_COMPONENT_COUNT:
                    return -12;
                case PNG_ERROR_STATE_NONE:
                default:
                    return 0;
            };
        }

        //call edge detection kernel
        void do_edge_detection()
        {
            edge_detection_on_gpu<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //call gray scale kernel
        void do_gray_scale()
        {
            gray_scale_on_gpu<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //call average pooling kernel
        void do_average_pooling()
        {
            average_pooling_on_gpu<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //call max pooling kernel
        void do_max_pooling()
        {
            max_pooling_on_gpu<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //call min pooling kernel
        void do_min_pooling()
        {
            min_pooling_on_gpu<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //call copy output kernel
        void do_copy_output_to_input()
        {
            copy_output_to_input<<<*( this->gridSize ), *( this->blockSize )>>>( this->ptr_image_data_on_gpu, this->ptr_output_data_on_gpu, this->image_width, this->image_height, this->used_component_count );
        }

        //run all the kernels in a single stream
        void run_kernels()
        {
            std::cout << "Running CUDA Kernel...\r\n";
            std::cout << "Copy data to GPU...\r\n";

            cudaMalloc( &( this->ptr_image_data_on_gpu ), this->element_count );
            cudaMalloc( &( this->ptr_output_data_on_gpu ), this->element_count );
            cudaMemcpy( this->ptr_image_data_on_gpu, this->all_pixel_values, this->element_count, cudaMemcpyHostToDevice );
            cudaMemcpy( this->ptr_output_data_on_gpu, this->output_pixel_values, this->element_count, cudaMemcpyHostToDevice );

            //loop over all kernel in the stream
            for( png_kernel_option_t kernel : this->png_kernel_options )
            {
                switch ( kernel )
                {
                    case PNG_EDGE_DETECTION:
                        do_edge_detection();
                        this->image_width -= 3;
                        this->image_height -= 3;
                        break;
                    case PNG_GRAY_SCALE:
                        do_gray_scale();
                        break;
                    case PNG_MAX_POOLING:
                        do_max_pooling();
                        this->image_width /= 2;
                        this->image_height /= 2;
                        break;
                    case PNG_MIN_POOLING:
                        do_min_pooling();
                        this->image_width /= 2;
                        this->image_height /= 2;
                        break;
                    case PNG_AVERAGE_POOLING:
                        do_average_pooling();
                        this->image_width /= 2;
                        this->image_height /= 2;
                        break;
                }

                do_copy_output_to_input();
            }

            cudaDeviceSynchronize();

            std::cout << "Copy data from GPU...\r\n";

            cudaMemcpy( this->all_pixel_values, this->ptr_image_data_on_gpu, this->element_count, cudaMemcpyDeviceToHost );
            cudaMemcpy( this->output_pixel_values, this->ptr_output_data_on_gpu, this->element_count, cudaMemcpyDeviceToHost );
            cudaFree( this->ptr_image_data_on_gpu );
            cudaFree( this->ptr_output_data_on_gpu );

            std::cout << "done!\r\n";
        }

        //write the image to the given filename
        void write_to_output( std::string output_file_name )
        {
            std::cout << "writing png to disk...\r\n";
            stbi_write_png( output_file_name.c_str(), this->image_width, this->image_height, this->used_component_count, this->output_pixel_values, ( this->used_component_count * this->image_width ) );
        }

        void print_data()
        {
            std::cout << " imagesize: " << this->image_width << "x" << this->image_height << "\r\n" << " component count: " << this->used_component_count << "\r\n\r\n";
        }
};

int main( int argc, char *argv[] )
{
    //init check
    if ( argc < 3 )
    {
        std::cout << "error: not enough arguments, expected minimum two\n\r";
        return -1;
    }

    //init
    std::string png_file_filename( argv[1] );
    
    png_image_data *input_file = new png_image_data::png_image_data( png_file_filename );
    if ( input_file->get_last_error() != 0 ) return -1;
    
    std::cout << "loaded .png file\r\n";
    input_file->print_data();
    
    //fill the buffer for calling all the kernels
    for ( int i = 2; i < argc; ++i )
    {
        std::string operation( argv[i] );

        if ( operation == "gray_scale" ) input_file->png_kernel_options.push_back( png_image_data::PNG_GRAY_SCALE );
        else if ( operation == "edge_detection" ) input_file->png_kernel_options.push_back( png_image_data::PNG_EDGE_DETECTION );
        else if ( operation == "min_pooling" ) input_file->png_kernel_options.push_back( png_image_data::PNG_MIN_POOLING );
        else if ( operation == "max_pooling" ) input_file->png_kernel_options.push_back( png_image_data::PNG_MAX_POOLING );
        else if ( operation == "average_pooling" ) input_file->png_kernel_options.push_back( png_image_data::PNG_AVERAGE_POOLING );
        else std::cout << "error: invalid operation " << operation << "\n\r";
    }
    input_file->run_kernels();

    input_file->print_data();
    input_file->write_to_output( "output_file.png" );

    //cleanup
    std::cout << "DONE\r\n";
    return 0;
}

