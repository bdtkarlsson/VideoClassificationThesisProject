import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.*;

/**
 * Created by bdtkarlsson on 2016-10-20.
 */
public class VideoClassificationThesisProject {


    private static final int video_height = 224;
    private static final int video_width = 224;
    private static final int channels = 3;
    private static final int minibatchsize = 16;
    private static final int nrOfCategories = 4;
    private static final String savedModelsPath = "saved_models";

    /*Early stopping training parameters*/
    private static final int maxEpochs = 500;
    private static final int maxHours = 200;
    private static final int maxEpochsWithoutImprovement = 5;

    /*Non-sequential data parameters*/
    private static final String[] allowedExtensions = {"bmp"};
    private static final String nonSeqDataPath = "video_data/nonsequential_data/data_1";
    private static final String nonSeqDataPath2 = "video_data/nonsequential_data/data_2";

    /*Sequential data parameters*/
    private static final int startFrame = 0;
    private static final int nrOfFramesPerVideo = 10;
    private static final String seqTrainingDataPath = "video_data/sequential_data/training_data";
    private static final String seqTestingDataPath = "video_data/sequential_data/testing_data";
    private static final String fileNameStandard = "ssportclip2_%d";

    public static void main(String[] args) {
        //evaluateVideoClips(false);
        //evaluateModelSeq();
       // evaluateModelNonSeq("saved_models2/bestModel.bin");
         //trainModel1();
        //trainModel2();
        //trainModel4();
        //trainModel3();
        //trainModel2();

    }

    private static void trainModel1() {
        MultiLayerConfiguration conf = NetworkModels.getModel1(video_height, video_width, channels, nrOfCategories);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1), new HistogramIterationListener(1));


        DataSetIterator[] data = null;
        try {
            data = DataLoader.getNonSequentialData(nonSeqDataPath,
                    allowedExtensions, video_height, video_width, channels, 64, 90, nrOfCategories);

        } catch (IOException e) {
            e.printStackTrace();
        }
        NetworkTrainer.earlyStoppingTrain(model, savedModelsPath, data[0], data[1], maxEpochs, maxHours, 5);

    }

    private static void trainModel2() {
        MultiLayerConfiguration conf = NetworkModels.getModel2(video_height, video_width, channels, nrOfCategories);

        MultiLayerNetwork model = null;
        DataSetIterator[] data = null;
        try {
            model = ModelHandler.loadModel("saved_models/model2it1.bin");
            model.setListeners(new ScoreIterationListener(1));
            data = DataLoader.getNonSequentialData(nonSeqDataPath2,
                    allowedExtensions, video_height, video_width, channels, minibatchsize, 90, nrOfCategories);
            PrintStream out = new PrintStream(new FileOutputStream("output.txt"));
            System.setOut(out);
        } catch (IOException e) {
            e.printStackTrace();
        }

        NetworkTrainer.earlyStoppingTrain(model, "saved_models2", data[0], data[1], maxEpochs, maxHours, 5);
    }

    private static void trainModel3() {
        MultiLayerConfiguration conf = NetworkModels.getModel3(video_height, video_width, channels, nrOfCategories,
                nrOfFramesPerVideo);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1), new HistogramIterationListener(1));

        DataSetIterator testingData = null, trainingData = null;
        try {
            testingData = DataLoader.getSequentialData(seqTestingDataPath, fileNameStandard, 0, 100, minibatchsize,
                    startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);

            trainingData = DataLoader.getSequentialData(seqTrainingDataPath, fileNameStandard, 0, 1260, minibatchsize,
                    110, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        NetworkTrainer.earlyStoppingTrain(model, savedModelsPath, trainingData, testingData, maxEpochs, maxHours,
                maxEpochsWithoutImprovement);
    }

    private static void trainModel4() {
        MultiLayerConfiguration conf = NetworkModels.getModel4(video_height, video_width, channels, nrOfCategories,
                nrOfFramesPerVideo);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        DataSetIterator testingData = null, trainingData = null;
        try {
            testingData = DataLoader.getSequentialData(seqTestingDataPath, fileNameStandard, 0, 100, minibatchsize,
                    startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);

            trainingData = DataLoader.getSequentialData(seqTrainingDataPath, fileNameStandard, 0, 1260, minibatchsize,
                    startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        NetworkTrainer.earlyStoppingTrain(model, savedModelsPath, trainingData, testingData, maxEpochs, maxHours,
                maxEpochsWithoutImprovement);
    }



    private static void evaluateModelSeq() {
        MultiLayerNetwork model = null;
        DataSetIterator testingData = null, trainingData = null;
        try {
            testingData = DataLoader.getSequentialData(seqTestingDataPath, fileNameStandard, 0, 168, minibatchsize,
                    startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);

            //trainingData = DataLoader.getSequentialData(seqTrainingDataPath, fileNameStandard, 0, 1260, minibatchsize,
            //       startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);
            model = ModelHandler.loadModel("saved_models/bestModel.bin");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Evaluation eval = NetworkEvaluator.evaluate(model, testingData, true);
        NetworkEvaluator.printAdvancedStats(eval, 4);
        System.out.println(eval.stats());
        // eval = NetworkEvaluator.evaluate(model, trainingData, labelMap, true);
        // System.out.println(eval.stats());
    }

    private static void evaluateModelNonSeq(String pathModel) {
        MultiLayerNetwork model = null;
        DataSetIterator[] data = null;
        try {
            data = DataLoader.getNonSequentialData(nonSeqDataPath,
                    allowedExtensions, video_height, video_width, channels, minibatchsize, 70, nrOfCategories);
            model = ModelHandler.loadModel(pathModel);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(model);
        Evaluation eval = NetworkEvaluator.evaluate(model, data[1], false);
        NetworkEvaluator.printAdvancedStats(eval, 4);
        System.out.println(eval.stats());
        eval = NetworkEvaluator.evaluate(model, data[0], false);
        NetworkEvaluator.printAdvancedStats(eval, 4);
        System.out.println(eval.stats());
    }

    private static void evaluateVideoClips(boolean seqData) {
        int[] classifiedVideos = new int[nrOfCategories];
        int[][] correctlyClassifiedVideos = new int[nrOfCategories][nrOfCategories];

        int[] classifiedFrames = new int[nrOfCategories];
        int[][] correctlyClassifiedFrames= new int[nrOfCategories][nrOfCategories];

        double[] tp = new double[nrOfCategories];
        double[] fp = new double[nrOfCategories];
        double[] tn = new double[nrOfCategories];
        double[] fn = new double[nrOfCategories];

        for(int i = 0; i < 1260; i++) {
            String path = "video_data/sequential_data/training_data/ssportclip2_" + i;
            File labelFile = new File(path + ".txt");
            BufferedReader br = null;
            int category = -1;
            try {
                br = new BufferedReader(new FileReader(path + ".txt"));
                String line = br.readLine();
                line = br.readLine();
                category = Integer.parseInt(line);
                Evaluation eval = null;
                if(seqData) {

                } else {
                    eval = NetworkEvaluator.evaluateVideoClipNonSeq(ModelHandler.loadModel("saved_models2/bestModel.bin"),
                            path + ".mp4", category, 0, 10, 10);
                }
                System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval.recall());

                fp[category] += eval.falsePositives().get(category);
                tp[category] += eval.truePositives().get(category);
                fn[category] += eval.falseNegatives().get(category);
                tn[category] += eval.trueNegatives().get(category);

                classifiedVideos[category]++;
                int mostClassifiedCategory = NetworkEvaluator.getMostClassifiedCategory(eval, category, nrOfCategories);
                correctlyClassifiedVideos[category][mostClassifiedCategory] ++;

                classifiedFrames[category] += 10;
                for(int j = 0; j < nrOfCategories; j++) {
                    if(j == category) {
                        correctlyClassifiedFrames[category][category] += eval.truePositives().get(category);
                    } else {
                        correctlyClassifiedFrames[category][j] += eval.falsePositives().get(j);
                    }
                }

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }


        for(int i = 0; i < nrOfCategories; i++) {
            System.out.println();
            System.out.println("CATEGORY: " + LabelMap.labelMap.get(i));


            System.out.println("Nr of classified Videos: " + classifiedVideos[i]);

            for(int j = 0; j < nrOfCategories; j++) {
                System.out.println("Nr of videos classified as " + LabelMap.labelMap.get(j) + ": " + correctlyClassifiedVideos[i][j]);
            }

            System.out.println("Nr of classified frames: " + classifiedFrames[i]);

            for(int j = 0; j < nrOfCategories; j++) {
                System.out.println("Nr of frames classified as " + LabelMap.labelMap.get(j) + ": " + correctlyClassifiedFrames[i][j]);
            }


        }

    }


}
