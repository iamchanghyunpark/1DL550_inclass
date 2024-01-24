# Lecture 3 Data-level-parallelism (SIMD) In-class codes

In this module we will look at two tasks.

## 1. Converting a color image into a grayscale image.
To do this, you will be working with a PPM image...

ADD ON INFORMATION THAT I HAD STARTED WRITING!


Run the `setup_img_links.sh` to setup symbolic links to the images that
I have uploaded into our project shared directory.

```
./setup_img_links.sh
```

Now, you are ready to start coding.
There are three parts of this program:

1. `grayscale_image.c`: This is the main file you will be working on.
Fill in the `grayscale_image_simd_{sse|avx|neon|neon_fp16}` functions.
2. `ppm.c` and `ppm.h`: This function holds everything to do with the PPM format.
Any reading of actual ppm files, any writing back into pgm files are done here.
We also cheat here. Usually PPM files have R, G, and B value for each pixel interleaved.
This is challenging to work with SIMD as we then need to work with swizzling
data in different parts of the vector register lane, etc.
Therefore, to make this simpler, I have organized the on-memory representation as
an array of Red pixels, another array of Green pixels and finally an array of blue pixels.
This will make SIMD processing much easier!
3. `common.c` and `common.h` At this point, it only holds timer related functions.

Now your implementatin of `grayscale_image_simd_{sse|avx|neon|neon_fp16}` merely
needs to read in each pixel and its three colors (R,G,B) values and calculate the
grayscale value. Use the following formula.

```
grayscale = R*0.21 + G*0.72 + B*0.07
```

Once you have implemented your version, say `avx`. You can run your program using the following method:

```
./grayscale_image avx IMG_5440.ppm
```

Then check the generated grayscale image that is in the PGM format. You can use `eog` that is installed on rackham.
You need to have x-forwarding set up to use this.

```
eog IMG_5440.pgm
```

Compare with the IMG\_5440.ppm (again using eog) and you should find that they are similar and PGM file is the gray-scale version of the PPM image.

If you are working from a Mac, instead of eog, you can use the Preview app to view it. From the terminal you can invoke the Preview app by running the following command:

```
open IMG_5440.pgm
```

Once you are ready and fairly convinced that your code is working well, you can run timing experiments by using the following command

```
make run_grayscale_x86 # if you are on an x86 machine, or rackham/snowy
make run_grayscale_arm # if you are working on your Mac with an Apple silicon
```

## 2. Monte-carlo Pi simulation
Generate many 100s of millions of random points with ranges [0,1] for both the x and y value.
Then go through each of the random point and measure the euclidean distance between the point and the origin.
If the distance is less-than or equal to 1, then you can say that this point lies within the radius of the circle.
(Assume that we have a circle `x^2 + y^2 = 1` and we are only looking at the first quadrant)
Now, you can get a portion out of the total points, how many lie within the distance of 1 and multiply by 4 and you get pi.
(Because the area of a circle of radius of 1 is pi).

Implement this in SIMD! I have implemented the non-vectorized version for you.

Oh and the file you will work on is `pi.c`.

Once you `make pi` you can run say your `neon` version using the following instruction.

```
./pi neon
```

If you have implemented both sse and avx, or just neon; you can compare non-vectorized (serial) and your vectorized version using the following command:

```
make run_pi_x86 # if you are on an x86 machine, or rackham/snowy
make run_pi_arm # if you are working on your Mac with an Apple silicon
```

Make sure to check that the Pi value printed for your SIMD version is relatively close to the Pi value from the serial version!
