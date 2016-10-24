import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * Method for evaluating a trained network
 */
public class NetworkEvaluator {

    private static final String tmpSeqFolder = "video_data/sequential_data/tmp_data/";
    private static final String tmpName = "video";

    /**
     *
     * @param model The network model used for evaluation
     * @param evaluationData The data used for evaluation
     * @param labelMap Used for labeling the categories
     * @return The evaluation result
     */
    public static Evaluation evaluate(MultiLayerNetwork model, DataSetIterator evaluationData,
                                      Map<Integer,String> labelMap, boolean sequentialData) {
        System.out.println("Starting evaluation...");
        Evaluation totalEvaluation = new Evaluation(labelMap);
        while (evaluationData.hasNext()) {
            DataSet dsTest = evaluationData.next();
            INDArray predicted = model.output(dsTest.getFeatureMatrix(), false);
            INDArray actual = dsTest.getLabels();
            if(sequentialData)
                totalEvaluation.evalTimeSeries(actual, predicted);
            else
                totalEvaluation.eval(actual, predicted);
        }
        return totalEvaluation;
    }

    public static Evaluation evaluateVideoClipSeq(MultiLayerNetwork model, String path, int category, int nrOfFrames,
                                                  Map<Integer, String> labelMap) {
        PrintWriter writer = null;
        DataSetIterator testData = null;
        /*Open file*/
        File f = new File(path);
        if(f.exists() && f.isFile()) {
            /*Create video and labels files in the tmp dir*/
            File videoFile = new File(tmpSeqFolder + tmpName + "_0.mp4");
            File labelFile =new File(tmpSeqFolder + tmpName + "_0.txt");
            try {
                /*Copy the input file to the temp folder*/
                FileUtils.copyFile(f, videoFile);

                /*Write category data to the label file in the temp folder*/
                writer = new PrintWriter(labelFile, "UTF-8");
                for(int j = 0; j < nrOfFrames; j++) {
                    writer.println(category);
                }
                writer.close();
                /*Load the data and the labels for testing*/
                testData = DataLoader.getSequentialData(tmpSeqFolder, tmpName + "_%d", 0, 1, 1, 0, nrOfFrames, 224, 224, 4);
            } catch (Exception e) {
                e.printStackTrace();
            }
            /*Evaluate the input file*/
            Evaluation eval = evaluate(model, testData, labelMap, true);
            /*Delete the tmp files*/
            labelFile.delete();
            videoFile.delete();
            return eval;
        }
        return null;
    }

    public static void printStats(Evaluation eval, Map<Integer, String> labelMap, int nrOfCategories) {

        double precision, recall, accuracy, f1;
        double precisionTotal = 0, recallTotal = 0, accuracyTotal = 0, f1Total = 0;
        double tp, fp, tn, fn;
        for(int i = 0; i < nrOfCategories; i++) {
            tp = eval.truePositives().get(i);
            fp = eval.falsePositives().get(i);
            tn = eval.trueNegatives().get(i);
            fn = eval.falseNegatives().get(i);
        /*    System.out.println("CATEGORY " + labelMap.get(i) + " RATES:" +
                    "\ntp: " + tp +
                    "\nfp: " + fp +
                    "\ntn: " + tn +
                    "\nfn: " + fn + "\n");*/
            precision = tp / (tp + fp);
            recall = tp / (tp + fn);
            accuracy = (tp + tn) / (tp + fp + tn + fn);
            f1 = 2 * ((precision * recall) / (precision + recall));
            System.out.println("CATEGORY " + labelMap.get(i) + " RESULTS:" +
                    "\nPrecision: " + precision +
                    "\nRecall: " + recall +
                    "\nAccuracy: " + accuracy +
                    "\nF1: " + f1 + "\n");
            precisionTotal += precision;
            accuracyTotal += accuracy;
            recallTotal += recall;
            f1Total += f1;
        }


        System.out.println("TOTALRESULTS:" +
                "\nPrecision: " + precisionTotal / nrOfCategories +
                "\nRecall: " + recallTotal / nrOfCategories +
                "\nAccuracy: " + accuracyTotal / nrOfCategories +
                "\nF1: " + f1Total / nrOfCategories + "\n");


    }

    public static void main(String[] args) {
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "ice hockey");
        labelMap.put(1, "soccer");
        labelMap.put(2, "basketball");
        labelMap.put(3, "american football");
        try {
            System.out.println(evaluateVideoClipSeq(ModelHandler.loadModel("saved_models/bestModel.bin"),
                    "video_data/sequential_data/testing_data/ssportclip2_90.mp4", 0, 100, labelMap).stats());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
