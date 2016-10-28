import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jcodec.api.FrameGrab8Bit;
import org.jcodec.api.JCodecException;
import org.jcodec.common.model.Picture8Bit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Method for evaluating a trained network
 */
public class NetworkEvaluator {

    private static final String tmpSeqFolder = "video_data/sequential_data/tmp_data/";
    private static final String tmpName = "video";
    private static final String tmpNonSeqFolder = "video_data/nonsequential_data/tmp_data/";

    /**
     *
     * @param model The network model used for evaluation
     * @param evaluationData The data used for evaluation
     * @return The evaluation result
     */
    public static Evaluation evaluate(MultiLayerNetwork model, DataSetIterator evaluationData, boolean sequentialData) {
        System.out.println("Starting evaluation...");
        Evaluation totalEvaluation = new Evaluation(LabelMap.labelMap);
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

    /**
     * Evaluate a video clip with a sequential model (e.g. LRCN)
     * @param model Model to be used in the classification
     * @param path Path to the video
     * @param category The correct category of the video
     * @param nrOfFrames Nr of frames from the video that should be classified
     * @return The Evaluation Stats
     */
    public static Evaluation evaluateVideoClipSeq(MultiLayerNetwork model, String path, int category, int startFrame, int nrOfFrames) {
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
                testData = DataLoader.getSequentialData(tmpSeqFolder, tmpName + "_%d", 0, 1, 1, startFrame, nrOfFrames, 224, 224, 4);
            } catch (Exception e) {
                e.printStackTrace();
            }
            /*Evaluate the input file*/
            Evaluation eval = evaluate(model, testData, true);
            /*Delete the tmp files*/
            labelFile.delete();
            videoFile.delete();
            return eval;
        }
        return null;
    }

    /**
     * Evaluate a video clip with a non-sequential model (e.g. CNN)
     * @param model Model to be used in the classification
     * @param path Path to the video
     * @param category The correct category of the video
     * @param nrOfFrames Nr of frames from the video that should be classified
     * @return The Evaluation Stats
     */
    public static Evaluation evaluateVideoClipNonSeq(MultiLayerNetwork model, String path, int category, int startFrame, int nrOfFrames, int frameJump) {
        PrintWriter writer = null;
        DataSetIterator testData = null;
        Evaluation eval = null;
        /*Open file*/
        File f = new File(path);
        if(f.exists() && f.isFile()) {
            try {
                for(int i = startFrame; i < (startFrame + nrOfFrames) * frameJump; i += frameJump) {
                    BufferedImage b = null;

                    Picture8Bit p = FrameGrab8Bit.getFrameFromFile(f, i);
                    b = AWTUtil.toBufferedImage8Bit(p);
                    File outputFile = new File(tmpNonSeqFolder + LabelMap.labelMap.get(category) + "/img_" + i + ".bmp");
                    ImageIO.write(b, "bmp", outputFile);
                }
                DataSetIterator data = DataLoader.getNonSequentialData(tmpNonSeqFolder + LabelMap.labelMap.get(category), new String[] {"bmp"}, 224,
                        224, 3, 10, 100, 4)[0];
                eval = evaluate(model, data, false);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (JCodecException e) {
                e.printStackTrace();
            }
            for(int i = startFrame; i < (startFrame + nrOfFrames) * frameJump; i += frameJump) {
                File fd = new File(tmpNonSeqFolder + LabelMap.labelMap.get(category) + "/img_" + i + ".bmp");
                fd.delete();
            }
        }

        return eval;
    }


    public static int getMostClassifiedCategory(Evaluation eval, int category, int nrOfCategories) {
        int tp = eval.truePositives().get(category);
        int bestCategory = category;
        int bestScore = tp;
        for(int i = 0; i < nrOfCategories; i++) {
            if(i != category) {
                if(eval.falsePositives().get(i) > bestScore) {
                    bestCategory = i;
                    bestScore = eval.falsePositives().get(i);
                }
            }
        }
        return bestCategory;
    }

    public static void printAdvancedStats(Evaluation eval, int nrOfCategories) {

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
            System.out.println("CATEGORY " + LabelMap.labelMap.get(i) + " RESULTS:" +
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


}
