import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.*;

/**
 * Author: Daniel Karlsson c11dkn@cs.umu.se
 */
public class VideoClassificationThesisProject {

    private static final int video_height = 168;
    private static final int video_width = 168;
    private static final int channels = 3;
    private static final int minibatchsize = 64;
    private static final int nrOfCategories = 11;
    private static final String savedModelsPath = "saved_models";

    /*Early stopping training parameters*/
    private static final int maxEpochs = 4;
    private static final int maxHours = 1500;
    private static final int maxEpochsWithoutImprovement = 5;

    /*Non-sequential data parameters*/
    private static final String[] allowedExtensions = {"bmp"};
    private static final String nonSeqDataPath = "video_data/nonsequential_data/data_1_it3";
    private static final String nonSeqDataPath2 = "video_data/nonsequential_data/data_2_it3";

    /*Sequential data parameters*/
    private static final int startFrame = 0;
    private static final int nrOfFramesPerVideo = 10;
    private static final String seqTrainingDataPath = "video_data/sequential_data/training_data3";
    private static final String seqTestingDataPath = "video_data/sequential_data/testing_data3";
    private static final String fileNameStandard = "sportclip_%d";

    public static void main(String[] args) {
        trainModel2();
    }

    private static void trainModel1() { // 6.5h 72 epochs
        MultiLayerConfiguration conf = NetworkModels.getModel1(video_height, video_width, channels, nrOfCategories);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));


        DataSetIterator[] data = null;
        try {
            data = DataLoader.getNonSequentialData(nonSeqDataPath,
                    allowedExtensions, video_height, video_width, channels, 64, 90, nrOfCategories);

        } catch (IOException e) {
            e.printStackTrace();
        }
        NetworkTrainer.earlyStoppingTrain(model, "saved_models", data[0], data[1], maxEpochs, maxHours, maxEpochsWithoutImprovement);

    }

    private static void trainModel2() { //it2: 85h 4 epochs it3: 44h 4 epochs
        MultiLayerConfiguration conf = NetworkModels.getModel2(video_height, video_width, channels, nrOfCategories);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));
        DataSetIterator[] data = null;
        try {

            data = DataLoader.getNonSequentialData(nonSeqDataPath2,
                    allowedExtensions, video_height, video_width, channels, 64, 90, nrOfCategories);
        } catch (IOException e) {
            e.printStackTrace();
        }

        NetworkTrainer.earlyStoppingTrain(model, "saved_models", data[0], data[1], maxEpochs, maxHours, maxEpochsWithoutImprovement);
    }

    private static void trainModel3() {
        MultiLayerConfiguration conf = NetworkModels.getModel3(video_height, video_width, channels, nrOfCategories);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        DataSetIterator testingData = null, trainingData = null;
        try {
            testingData = DataLoader.getSequentialData(seqTestingDataPath, fileNameStandard, 0, 462, minibatchsize,
                    startFrame, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);

            trainingData = DataLoader.getSequentialData(seqTrainingDataPath, fileNameStandard, 0, 3465, minibatchsize,
                    110, nrOfFramesPerVideo, video_height, video_width, nrOfCategories);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        NetworkTrainer.earlyStoppingTrain(model, savedModelsPath, trainingData, testingData, maxEpochs, maxHours,
                maxEpochsWithoutImprovement);
    }


    private static void evaluateVideoClips(boolean seqData, int nrOfFrames, int frameJump) {
        int[] classifiedVideos = new int[nrOfCategories];
        int[][] correctlyClassifiedVideos = new int[nrOfCategories][nrOfCategories];
        int[] classifiedFrames = new int[nrOfCategories];
        int[][] correctlyClassifiedFrames= new int[nrOfCategories][nrOfCategories];

        /*Load model*/
        MultiLayerNetwork seqModel = null;
        MultiLayerNetwork nonSeqModel = null;
        try {
            if(seqData) {
                seqModel = ModelHandler.loadModel("saved_models/model3it3.bin");
            } else {
                nonSeqModel = ModelHandler.loadModel("saved_models/model2it3b.bin");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i = 0; i < 3465; i++) {
            String path = "video_data/sequential_data/training_data3/sportclip_" + i;
            BufferedReader br = null;
            int category = -1;
            try {
                br = new BufferedReader(new FileReader(path + ".txt"));
                String line = br.readLine();
                line = br.readLine();
                category = Integer.parseInt(line);
                Evaluation eval = null;
                Evaluation eval2 = null;
                Evaluation eval3 = null;
                Evaluation eval4 = null;
                Evaluation eval5 = null;
                if(seqData) {
                    eval = NetworkEvaluator.evaluateVideoClipSeq(seqModel,
                            path + ".mp4", category, 0, 10, nrOfCategories);
                    eval2 = NetworkEvaluator.evaluateVideoClipSeq(seqModel,
                            path + ".mp4", category, 30, 10, nrOfCategories);
                    eval3 = NetworkEvaluator.evaluateVideoClipSeq(seqModel,
                            path + ".mp4", category, 50, 10, nrOfCategories);
                    eval4 = NetworkEvaluator.evaluateVideoClipSeq(seqModel,
                            path + ".mp4", category, 70, 10, nrOfCategories);
                    eval5 = NetworkEvaluator.evaluateVideoClipSeq(seqModel,
                            path + ".mp4", category, 90, 10, nrOfCategories);
                } else {
                    eval = NetworkEvaluator.evaluateVideoClipNonSeq(nonSeqModel,
                            path + ".mp4", category, 0, nrOfFrames, frameJump, nrOfCategories);
                }

                System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval.recall());
                if(seqData) {
                    System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval2.recall());
                    System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval3.recall());
                    System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval4.recall());
                    System.out.println("Video " + i + ", " + LabelMap.labelMap.get(category) + ": " + eval5.recall());
                }

                classifiedVideos[category]++;

                if(seqData) {
                    int[] mostClassifiedCategory = new int[5];
                    mostClassifiedCategory[0] = getMostClassifiedCategory(eval, category, nrOfCategories);
                    mostClassifiedCategory[1] = getMostClassifiedCategory(eval2, category, nrOfCategories);
                    mostClassifiedCategory[2] = getMostClassifiedCategory(eval3, category, nrOfCategories);
                    mostClassifiedCategory[3] = getMostClassifiedCategory(eval4, category, nrOfCategories);
                    mostClassifiedCategory[4] = getMostClassifiedCategory(eval5, category, nrOfCategories);
                    correctlyClassifiedVideos[category][getPopularElement(mostClassifiedCategory)]++;
                } else{
                    int mostClassifiedCategory = getMostClassifiedCategory(eval, category, nrOfCategories);
                    correctlyClassifiedVideos[category][mostClassifiedCategory]++;
                }

                classifiedFrames[category] += nrOfFrames;
                for(int j = 0; j < nrOfCategories; j++) {
                    if(j == category) {
                        correctlyClassifiedFrames[category][category] += eval.truePositives().get(category);
                        if(seqData) {
                            correctlyClassifiedFrames[category][category] += eval2.truePositives().get(category);
                            correctlyClassifiedFrames[category][category] += eval3.truePositives().get(category);
                            correctlyClassifiedFrames[category][category] += eval4.truePositives().get(category);
                            correctlyClassifiedFrames[category][category] += eval5.truePositives().get(category);
                        }
                    } else {
                        correctlyClassifiedFrames[category][j] += eval.falsePositives().get(j);
                        if(seqData) {
                            correctlyClassifiedFrames[category][j] += eval2.falsePositives().get(j);
                            correctlyClassifiedFrames[category][j] += eval3.falsePositives().get(j);
                            correctlyClassifiedFrames[category][j] += eval4.falsePositives().get(j);
                            correctlyClassifiedFrames[category][j] += eval5.falsePositives().get(j);
                        }
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

    private static int getPopularElement(int[] a) {
        int count = 1, tempCount;
        int popular = a[0];
        int temp = 0;
        for (int i = 0; i < (a.length - 1); i++)
        {
            temp = a[i];
            tempCount = 0;
            for (int j = 1; j < a.length; j++)
            {
                if (temp == a[j])
                    tempCount++;
            }
            if (tempCount > count)
            {
                popular = temp;
                count = tempCount;
            }
        }
        return popular;
    }

    private static void evaluateDemo(String[] args) {

        if(args.length >= 1) {

            String videoPath = args[0];
            MultiLayerNetwork model = null;
            System.out.println("Loading model...");
            try {
                model = ModelHandler.loadModel("model3it2.bin");
            } catch (IOException e) {
                System.err.println("ERROR: No such model exist, " + "model2it2.bin");
                return;
            }
            System.out.println("Model Loaded.");

            Evaluation eval = null;

            System.out.println("Start evaluation...");
            try {
                eval = NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 0, 10, 11);
                eval.merge(NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 100, 10, 11));
                eval.merge(NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 500, 10, 11));
                eval.merge(NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 1000, 10, 11));
                eval.merge(NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 1500, 10, 11));
                eval.merge(NetworkEvaluator.evaluateVideoClipSeq(model, videoPath, 1, 2000, 10, 11));
            } catch(NullPointerException e) {
                System.err.println("ERROR: No such video, " + videoPath);
                return;
            }
            System.out.println("Evaluation results:");
            for(int i = 0; i < 11; i++) {
                System.out.print("Percentage of frames predicted to be " + LabelMap.labelMap.get(i) + ": ");
                if(eval.truePositives().get(i) != 0) {
                    System.out.println((int) (eval.truePositives().get(i) / 0.6) + "%");
                } else {
                    System.out.println((int) (eval.falsePositives().get(i) / 0.6) + "%");
                }
            }

        } else {
            System.err.println("ERROR: No input");
        }
    }
}
