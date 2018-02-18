import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.SET;
import edu.princeton.cs.algs4.SeparateChainingHashST;

import java.io.File;
import java.util.StringTokenizer;

public class ExtractAmenities {

    private File test; // csv with 28 columns
    private File train; // csv with 29 columns (1 extra for log_price)

    private Dataset testDataset;
    private Dataset trainDataset;

    private SET<String> uniqueAmenities;

    private class Dataset {
        private File file;
        private SeparateChainingHashST<Integer, SET<String>> amenities;

        private Dataset(File file) {
            this.file = file;
            amenities = new SeparateChainingHashST<>();
        }
    }

    public ExtractAmenities(File testFile, File trainFile) {

        uniqueAmenities = new SET<>();

        testDataset = new Dataset(testFile);
        trainDataset = new Dataset(trainFile);

        readFile(testDataset);
        readFile(trainDataset);
    }

    public static void main(String[] args) {
        if (args.length != 2) throw new IllegalArgumentException("Usage: ExtractAmenities [testFile] [trainFile]");

        File testFile = new File(args[0]);
        File trainFile = new File(args[1]);

        ExtractAmenities extractAmenities = new ExtractAmenities(testFile, trainFile);
    }


    private void readFile(Dataset dataset) {
        In in = new In(dataset.file);

        String line;
        int c = 0;
        while (in.hasNextLine()){
            line = in.readLine();
            parseLine(dataset, line);
            c++;
        }
    }

    private void parseLine(Dataset dataset, String line) {
        int beginAt = 0;
        int endAt = 0;

        Integer key = Integer.parseInt(line.split(",")[0]);

        int pos = 0;
        while (line.charAt(pos) != '{') {
            pos++;
        }
        beginAt = pos;

        while (line.charAt(pos) != '}') {
            pos++;
        }
        endAt = pos;

        assert beginAt < endAt;

        String sub = line.substring(beginAt,endAt);
        sub.replace("\"","");
        sub.toLowerCase();
        StringTokenizer st = new StringTokenizer(",");

        SET<String> listingAmenities = new SET<>();
        while (st.hasMoreTokens()) {
            String amenity = st.nextToken();
            uniqueAmenities.add(amenity);
            listingAmenities.add(amenity);
        }
        dataset.amenities.put(key, listingAmenities);
    }

    private void oneHotEncode(Dataset dataset) {
        int n = dataset.amenities.size();
        int k = uniqueAmenities.size();

        int[][] output = new int[n][k + 1];

        int c = 0;
        for (Integer key : dataset.amenities.keys()) {
            SET<String> listingAmenities = dataset.amenities.get(key);

        }
    }
}
