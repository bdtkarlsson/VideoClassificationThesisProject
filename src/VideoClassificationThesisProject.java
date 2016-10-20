import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

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
    private static final String nonSeqTrainingDataPath = "video_data/nonsequential_data/training_data";
    private static final String nonSeqTestingDataPath = "video_data/nonsequential_data/testing_data";

    /*Sequential data parameters*/
    private static final int startFrame = 0;
    private static final int nrOfFramesPerVideo = 10;
    private static final String seqTrainingDataPath = "video_data/sequential_data/training_data";
    private static final String seqTestingDataPath = "video_data/sequential_data/testing_data";
    private static final String fileNameStandard = "ssportclip2_%d";

    public static void main(String[] args) {
       // trainModel3();
        evaluateModelSeq();
    }

    private static void trainModel1() {
        MultiLayerConfiguration conf = NetworkModels.getModel1(video_height, video_width, channels, nrOfCategories);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));


        DataSetIterator testingData = null, trainingData = null;
        try {
            testingData = DataLoader.getNonSequentialData(nonSeqTestingDataPath,
                    allowedExtensions, video_height, video_width, channels, minibatchsize, 10, true, nrOfCategories);
            trainingData = DataLoader.getNonSequentialData(nonSeqTrainingDataPath,
                    allowedExtensions, video_height, video_width, channels, minibatchsize, 10, true, nrOfCategories);
        } catch (IOException e) {
            e.printStackTrace();
        }

        NetworkTrainer.earlyStoppingTrain(model, savedModelsPath, trainingData, testingData, maxEpochs, maxHours,
                maxEpochsWithoutImprovement);
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
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "ice hockey");
        labelMap.put(1, "soccer");
        labelMap.put(2, "basketball");
        labelMap.put(3, "american football");
        Evaluation eval = NetworkEvaluator.evaluate(model, testingData, labelMap, true);
        NetworkEvaluator.printStats(eval, labelMap, 4);
        System.out.println(eval.stats());
       // eval = NetworkEvaluator.evaluate(model, trainingData, labelMap, true);
        // System.out.println(eval.stats());
    }
}
