import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by bdtkarlsson on 2016-10-27.
 */
public class LabelMap {

    public static final Map<Integer, String> labelMap = new HashMap();

    public static final ArrayList<String> labels = new ArrayList<String>();

    static {
        labels.add(0,"icehockey");
        labels.add(1,"soccer");
        labels.add(2,"basketball");
        labels.add(3,"football");

        labelMap.put(0, "icehockey");
        labelMap.put(1, "soccer");
        labelMap.put(2, "basketball");
        labelMap.put(3, "american football");
    }

}
