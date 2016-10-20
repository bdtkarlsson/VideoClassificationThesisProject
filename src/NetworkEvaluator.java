import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Method for evaluating a trained network
 */
public class NetworkEvaluator {

    /**
     *
     * @param model The network model used for evaluation
     * @param evaluationData The data used for evaluation
     * @param labelMap Used for labeling the categories
     * @return The evaluation result
     */
    public static Evaluation evaluate(MultiLayerNetwork model, DataSetIterator evaluationData,
                                      Map<Integer,String> labelMap) {
        Evaluation totalEvaluation = new Evaluation(labelMap);
        while (evaluationData.hasNext()) {
            DataSet dsTest = evaluationData.next();
            INDArray predicted = model.output(dsTest.getFeatureMatrix(), false);
            INDArray actual = dsTest.getLabels();
            totalEvaluation.eval(actual, predicted);
        }
        return totalEvaluation;
    }

    public static Evaluation evaluateVideoClip(MultiLayerNetwork model, String path, boolean sequentialModel) {
        return null;
    }

}
