package network.math;

import java.util.Random;

/**
 * Global random number generator
 * For easy use of setting the seed
 * For all random numbers
 */
public class MyRand {
    private static Random myRand = new Random();

    public static void initialize(int seed){
        myRand = new Random(seed);
    }

    public static void initialize(){
        myRand = new Random();
    }

    public static int randInt(int bound){
        return myRand.nextInt(bound);
    }

    public static float randFloat(){
        return myRand.nextFloat();
    }

    public static float randNormal() {
        return (float) myRand.nextGaussian();
    }

}
