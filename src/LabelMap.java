import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * label map and label list used in the evaluation of the network and the loading of non-sequential data
 */
public class LabelMap {

    public static final Map<Integer, String> labelMap = new HashMap();

    public static final ArrayList<String> labels = new ArrayList<String>();

    static {
        labels.add(0,"icehockey");
        labels.add(1,"soccer");
        labels.add(2,"basketball");
        labels.add(3,"football");

        for(int i = 0; i < labels.size(); i++) {
            labelMap.put(i, labels.get(i));
        }

    }

}
