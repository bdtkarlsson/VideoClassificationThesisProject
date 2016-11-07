import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.concurrent.TimeUnit;

/**
 * Methods for training a network model. The early stopping training stops depending on 3 different factors: If the number
 * of epochs is too high, if too much time has been spent on training or if the performance has not improved for N epochs.
 */
public class NetworkTrainer {


    /**
     * Trains a model with the provided data for the selected number of epochs
     *
     * @param model The model to be trained
     * @param trainData The training data
     * @param nrOfEpochs The number of training epochs
     * @return The trained network
     */
    public static MultiLayerNetwork train(MultiLayerNetwork model, DataSetIterator trainData, int nrOfEpochs) {
        for(int i = 0; i < nrOfEpochs; i++) {
            while(trainData.hasNext()) {
                model.fit(trainData.next());
            }
            trainData.reset();
        }
        return model;
    }

    /**
     * Trains the model with the provided data until one of the termination conditions are met. The conditions provided
     * are max hours, max epochs and max epochs without improvement. The model will be constantly evaluated with testing
     * data to be able to check the third termination condition. The best model will constantly be saved during
     * training.
     * @param model The model to be trained
     * @param modelSavePath Path to where the best model will be saved
     * @param trainData The training data
     * @param testData The testing data
     * @param maxEpochs Termination condition: max epochs
     * @param maxHours Termination condition: max hours
     * @param maxEpochsWithoutImprovement Termination condition: max epochs without improvement
     * @return The trained model (best model)
     */
    public static MultiLayerNetwork earlyStoppingTrain(MultiLayerNetwork model, String modelSavePath,
                                                       DataSetIterator trainData, DataSetIterator testData,
                                                       int maxEpochs, int maxHours, int maxEpochsWithoutImprovement) {

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs),
                        new ScoreImprovementEpochTerminationCondition(maxEpochsWithoutImprovement))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(maxHours, TimeUnit.HOURS))
                .scoreCalculator(new DataSetLossCalculator(testData, true))

                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(modelSavePath))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainData);
        //Conduct early stopping training:
        EarlyStoppingResult result = trainer.fit();

        return (MultiLayerNetwork) result.getBestModel();
    }




}
