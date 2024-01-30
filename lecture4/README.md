# Lecture 4 Mass Parallelism: CUDA programming - in-class session

In this module I have prepared three tasks. (But two of them you are familiar with!)

BTW, run `make` when you want to build.
```
make grayscale_image # if you want to generate code for the first task
make pi # if you want to build the second task
make matmult # if you want to build the final task
```
Once you have filled in all three `.cu`j files, you should be able to simply just run `make` and the Makefile will build all three programs.

## 1. Converting a color image into a grayscale image. (Yet again!)

For the SIMD implementation, let's admit it. We cheated!
Chang Hyun manipulated the data layout so that it would be favorable for SIMD.
Also while at it, Chang Hyun *conviniently* changed the type from unsigned chars (1B integers) to FP32 types.

No more of this cheating! Now we will do this properly.
We will be working with unsigned char (1B) values directly in the CUDA kernel.
We will be using the SIMD unfavorable, or actual data representation of PPM images! RGBRGBRGB...
And we will find that with CUDA, it is actually really simple to do!

For this work, unless you have a CUDA-capable NVIDIA GPU on your laptop (which I doubt ;P) you will most likely work on UPPMAX.
If you are UPPMAX, run the `setup_img_links.sh` to setup symbolic links to the images that
I have uploaded into our project shared directory.

```
./setup_img_links.sh
```

Now, you are ready to start coding.
There are three parts of this program:

Open the grayscale\_image.cu and check the comments that I have placed for you.
Using the comments, fill in the code, compile and run! Remember that the formula to get a grayscale value is as follows:

```
grayscale = R*0.21 + G*0.72 + B*0.07
```

Once you have implemented your version, you can run your program using the following method:

```
./grayscale_image IMG_5440.ppm
#or
./grayscale_image *.ppm
```

Then check the generated grayscale image that is in the PGM format. You can use `eog` that is installed on rackham.
You need to have x-forwarding set up to use this.

```
eog IMG_5440.pgm
```

Compare with the IMG\_5440.ppm (again using eog) and you should find that they are similar and PGM file is the gray-scale version of the PPM image.

## 2. Monte-carlo Pi simulation

Another task that you are familiar with! Open up the `pi.cu` file.
This time, we will need to focus on allocating host and device memory; copying data from host to device using cuda commands (e.g. `cudaMemcpy()`)
Dividing the job up into blocks and threads per blocks, invoking the kernel; and copying the data back from the device memory.

You will also need to fill in some blanks in the kernel.
One blanking out moment I had while preparing this was 'how do we sum up all the instances from all the threads that have points that lie within the distance of 1.0?
This is challenging and I ended up using 'reductions' to do this.
Check the code if you are interested.

Once you can build your program run it!
```
./pi
```

## 3. Matrix Multiplication
A classical task for GPGPU programming. Open up the `matmult.cu` file.
Create a cuda program to multiply to matrices!
Fill in the blanks (where I have conviniently for you, provided comments. Maybe too much comments... I hope the task being too easy does not bore you too much :)

Happy hacking!

