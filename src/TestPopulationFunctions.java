import java.util.Arrays;

public class TestPopulationFunctions {

    public static void main(String[] args)
    {
        int[] data = {5, 6, 4, 3, 1, 2, 6};
        int n = 4;
        int[] topN = Population.sortTopN(data, n);
        System.out.println(Arrays.toString(data));
        System.out.println(Arrays.toString(topN));
    }
}
