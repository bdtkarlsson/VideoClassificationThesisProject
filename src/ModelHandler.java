import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;
import java.io.IOException;

/**
 * Methods for saving and loading trained network models.
 */
public class ModelHandler {

    /**
     * Method that saves a network model into a file. Used for saving trained network models.
     * @param model The model to be saved
     * @param fileName The desired filename
     * @throws IOException
     */
    public static void saveModel(MultiLayerNetwork model, String fileName) throws IOException {
        File file = new File(fileName);
        ModelSerializer.writeModel(model, file, true);
    }

    /**
     * Method for loading a network model. Used for retrieving a trained network.
     * @param fileName The name of the file containing the model
     * @return The loaded model
     * @throws IOException
     */
    public static MultiLayerNetwork loadModel(String fileName) throws IOException {
        File file = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(file);
        return model;
    }
}
