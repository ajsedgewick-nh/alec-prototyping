import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.Collection;
import java.util.Comparator;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.karaf.shell.api.action.Action;
import org.apache.karaf.shell.api.action.Command;
import org.apache.karaf.shell.api.action.Option;
import org.apache.karaf.shell.api.action.lifecycle.Service;
import org.opennms.alec.datasource.api.Alarm;
import org.opennms.alec.datasource.api.InventoryObject;
import org.opennms.alec.datasource.api.Situation;
import org.opennms.alec.datasource.jaxb.JaxbUtils;
import org.opennms.alec.driver.test.TestDriver;
import org.opennms.alec.engine.api.Engine;
import org.opennms.alec.engine.api.EngineFactory;
import org.opennms.alec.engine.cluster.AbstractClusterEngine;
import org.opennms.alec.engine.cluster.AlarmInSpaceTime;
import org.opennms.alec.engine.cluster.CEEdge;
import org.opennms.alec.engine.cluster.CEVertex;
import org.opennms.alec.engine.deeplearning.InputVector;
import org.opennms.alec.engine.deeplearning.OutputVector;
import org.opennms.alec.engine.deeplearning.Vectorizer;
import org.opennms.alec.engine.dbscan.DBScanEngine;
import org.opennms.alec.engine.dbscan.AlarmInSpaceTimeDistanceMeasure;

import com.codahale.metrics.MetricRegistry;

import edu.uci.ics.jung.graph.Graph;

// We're taking org.opennms.features.deeplearning.shell.Vectorize.java and making it work on the command line
// and have it output alarm IDs so we can use it as a distance matrix
// USAGE: CmdVector input_alarms.xml input_inventory.xml input_situations.xml output.csv
public class CmdVector{//} implements Action{


    private static String alarmsIn;
    private static String inventoryIn;
    private static String situationsIn;
    private static String csvOut;
    private static long tickRes = 268000000; //24 * 60 * 60 * 1000; // tick res is 1 day
    //private static Map<String, long[]> alarmTimes = new LinkedHashMap<String, long[]>();

    public static void main(String args[]){
        alarmsIn = args[0];
        inventoryIn = args[1];
        situationsIn = args[2];
        csvOut = args[3];
 
        try {execute();}
        catch (Exception e){
        	System.out.println("Execute failed " + e);
            e.printStackTrace();
        }

    }

    //@Override
    //public Object execute() throws Exception{
    public static Object execute() throws Exception{
        final List<Alarm> alarms = JaxbUtils.getAlarms(Paths.get(alarmsIn));
        final List<InventoryObject> inventory = JaxbUtils.getInventory(Paths.get(inventoryIn));
        final Set<Situation> situations = JaxbUtils.getSituations(Paths.get(situationsIn));

        // Need to get max and min alarm times here because driver only takes latest per tick
        // TODO: consiter severity/cleared?
        /*for (Alarm a1: alarms){
        	if (!alarmTimes.contains(a1.getId())){
        		alarmTimes.put(a1.getId(), new long[] {a1.getTime(), a1.getTime()});
        	} else {
        		long curTime = a1.getTime();
            	long[] curSpan = alarmTimes.get(a1.getId());
            	if (curSpan[0] > curTime)
            	    curSpan[0] = curTime;
            	if (curSpan[1] < curTime)
            		curSpan[1] = curTime;
        	}

            /*
        	for (Alarm a2: alarms){
        		if ((a1.getId().compareTo(a2.getId()) == 0) &&
          			(a1.getTime() != a2.getTime()))
       		        System.out.printf("Found ID match with different times: %s,%d vs. %s,%d\n", a1.getId(),
                     	a1.getTime(), a2.getId(), a2.getTime());
        	}
        }*/

        final Path path = Paths.get(csvOut);
        System.out.printf("Writing to: %s\n", path);
        try (
                BufferedWriter writer = Files.newBufferedWriter(Paths.get(csvOut));
                CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT
                        .withHeader(getHeader()));
        ) {
            streamVectors(inventory, alarms, situations, v -> {
                try {
                    csvPrinter.printRecord(toRecordValues(v));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
            csvPrinter.flush();
        }
        System.out.println("Done.");

        return null;
    }

    private static class MyEngine extends AbstractClusterEngine {
        private final Consumer<MyOutputVector> consumer;
        private final Set<Situation> situations;
        private final Map<String,String> alarmIdToSituationId = new LinkedHashMap<>();
        private Vectorizer vectorizer;

        private MyEngine(Set<Situation> situations, Consumer<MyOutputVector> consumer) {
            super(new MetricRegistry());
            this.situations = Objects.requireNonNull(situations);
            this.consumer = Objects.requireNonNull(consumer);

            // Index the alarms by situation id
            for (Situation s : situations) {
                for (Alarm a : s.getAlarms()) {
                    alarmIdToSituationId.put(a.getId(), s.getId());
                }
            }
        }

        @Override
        public void onInit() {
            vectorizer = new Vectorizer(getGraphManager(), this);
        }

        @Override
        public List<Cluster<AlarmInSpaceTime>> cluster(long timestampInMillis, Graph<CEVertex, CEEdge> g) {
            final List<AlarmInSpaceTime> alarms = getAlarmsInSpaceTime(g);

            final AlarmInSpaceTimeDistanceMeasure distanceMeasure = new AlarmInSpaceTimeDistanceMeasure(this, DBScanEngine.DEFAULT_ALPHA, DBScanEngine.DEFAULT_BETA);

            for (AlarmInSpaceTime a1 : alarms) {
                for (AlarmInSpaceTime a2 : alarms) {
                    // diff comparison assuming alarm distances are symmetric, lex comparison of Id for now
                    /*if (a1.getAlarmId().compareTo(a2.getAlarmId()) >= 0) {
                	    continue;
                	}*/
                	// this finds lots of mismatches because events are considered seperately
                	if (a1.getAlarmId().compareTo(a2.getAlarmId()) >= 0){
                		if ((a1.getAlarmId().compareTo(a2.getAlarmId()) == 0) &&
                			(a1.getAlarmTime() != a2.getAlarmTime()))
                		        System.out.printf("Found ID match with different times: %s,%d vs. %s,%d\n", a1.getAlarmId(),
                		        	a1.getAlarmTime(), a2.getAlarmId(), a2.getAlarmTime());
                        continue;
                    }

                    final InputVector inputVector = vectorizer.vectorize(a1, a2);
                    final OutputVector outputVector = OutputVector.builder()
                            .inputVector(inputVector)
                            .areAlarmsRelated(areAlarmsCurrentlyRelated(a1, a2, timestampInMillis))
                            .build();
                    consumer.accept(MyOutputVector.builder()
                    	    .a1(a1) //.getAlarmId())
                    	    .a2(a2)
                    	    .outputVector(outputVector)
                    	    .stdist(distanceMeasure.compute(a1.getPoint(), a2.getPoint()))
                    	    .build());

                }
            }
            return Collections.emptyList();
        }

        private List<AlarmInSpaceTime> getAlarmsInSpaceTime(Graph<CEVertex,CEEdge> g) {
            // This is how Vectorize collects alarms
            /*List<AlarmInSpaceTime> alarmsInSpaceAndTime = new LinkedList<>();
            for (CEVertex v : g.getVertices()) {
                for (Alarm a : v.getAlarms()) {
                    alarmsInSpaceAndTime.add(new AlarmInSpaceTime(v,a));
                }
            }*/
            

            // This is how DBScan engine collects alarms...
            // Ensure the points are sorted in order to make sure that the output of the clusterer is deterministic
            // OPTIMIZATION: Can we avoid doing this every tick?
            
            List<AlarmInSpaceTime> alarmsInSpaceAndTime = g.getVertices().stream()
                .map(v -> v.getAlarms().stream()
                        .map(a -> new AlarmInSpaceTime(v, a))
                        .collect(Collectors.toList()))
                .flatMap(Collection::stream)
                .sorted(Comparator.comparing(AlarmInSpaceTime::getAlarmTime).thenComparing(AlarmInSpaceTime::getAlarmId))
                .collect(Collectors.toList());
            
                
            /*
            Map<String, AlarmInSpaceTime> filterAlarmsInSpaceAndTime = new HashMap<>();
            for (AlarmInSpaceTime a : alarmsInSpaceAndTime){
            	if(!a.getAlarm().isClear()){
                    filterAlarmsInSpaceAndTime.put(a.getAlarmId(), a);
            	}
            }

            alarmsInSpaceAndTime = filterAlarmsInSpaceAndTime.values().stream()
                .sorted(Comparator.comparing(AlarmInSpaceTime::getAlarmTime).thenComparing(AlarmInSpaceTime::getAlarmId))
                .collect(Collectors.toList());
            */
            // uses TestDriver methods to collect and reduce alarms
            //alarmsInSpaceAndTime = TestDriver.timeSortAlarms(alarmsInSpaceAndTime);
            //alarmsInSpaceAndTime = TestDriver.reduceAlarms(alarmsInSpaceAndTime);
            return alarmsInSpaceAndTime;
        }

        private boolean areAlarmsCurrentlyRelated(AlarmInSpaceTime a1, AlarmInSpaceTime a2, long timestampInMillis) {
            // TODO: We should improve this to consider time as well
            final String s1 = alarmIdToSituationId.get(a1.getAlarmId());
            final String s2 = alarmIdToSituationId.get(a2.getAlarmId());
            if (s1 == null || s2 == null) {
                return false;
            }
            return Objects.equals(s1,s2);
        }
    }


    public static class MyOutputVector{
    	private final AlarmInSpaceTime a1;
    	private final AlarmInSpaceTime a2;
        private final OutputVector outputVector;
        private final double stdist;

        private MyOutputVector(Builder builder){
            this.outputVector = builder.outputVector;
        	this.a1 = builder.a1;
        	this.a2 = builder.a2;
        	this.stdist = builder.stdist;
        }

        public OutputVector getOutputVector(){
           	return outputVector;
        }

        public AlarmInSpaceTime getA1(){
           	return a1;
        }

        public AlarmInSpaceTime getA2(){
            return a2;
        }


        public double getStdist(){
            return stdist;
        }

        public static Builder builder() {
            return new Builder();
        }

    	//@Override
    	public static class Builder{
            private AlarmInSpaceTime a1;
            private AlarmInSpaceTime a2;

            private OutputVector outputVector;
            private double stdist;

            public Builder a1(AlarmInSpaceTime a1){
            	this.a1 = a1;
            	return this;
            }

            public Builder a2(AlarmInSpaceTime a2){
            	this.a2 = a2;
            	return this;
            }


            public Builder outputVector(OutputVector outputVector){
            	this.outputVector = outputVector;
            	return this;
            }
            
            public Builder stdist(double stdist){
            	this.stdist = stdist;
            	return this;
            }

            public MyOutputVector build(){
            	Objects.requireNonNull(a1, "alarm 1 must not be null");
            	Objects.requireNonNull(a2, "alarm 2 must not be null");
   	            Objects.requireNonNull(outputVector, "outputVector must not be null.");
   	            Objects.requireNonNull(stdist, "stdist must not be null.");
            	return new MyOutputVector(this);
            }

    	}
    }

    private static class MyEngineFactory implements EngineFactory {
        private final MyEngine engine;

        public MyEngineFactory(MyEngine engine) {
            this.engine = Objects.requireNonNull(engine);
        }

        @Override
        public String getName() {
            return "test";
        }

        @Override
        public Engine createEngine(MetricRegistry metrics) {
            return engine;
        }
    }

    private static void streamVectors(List<InventoryObject> inventory, List<Alarm> alarms, Set<Situation> situations, Consumer<MyOutputVector> consumer) {
        // Extend the cluster engine
        final MyEngine engine = new MyEngine(situations, consumer);

        engine.setTickResolutionMs(tickRes);

        // Use the test driver
        final MyEngineFactory factory = new MyEngineFactory(engine);
        final TestDriver driver = TestDriver.builder()
                .withEngineFactory(factory)
                .withVerboseOutput()
                .build();

        // On tick, the engine will perform a pairwise comparison of all active alarms, convert these
        // to vectors and feed it to the consumer
        driver.run(alarms, inventory);
    }

    private static String[] getHeader() {
        return new String[]{
        	    "id_a",
        	    "id_b",
                "type_a",
                "type_b",
                "time_a",
                "time_b",
                "same_instance",
                "same_parent",
                "share_ancestor",
                "time_delta_ms",
                "distance_on_graph",
                "distance_in_st",
                "io_id_similarity",
                "io_label_similarity",
                "related"
        };
    }

    /**
     * Used for CSV output.
     */
    private static Iterable<?> toRecordValues(MyOutputVector mv) {
    	final OutputVector ov = mv.getOutputVector();
        final InputVector v = ov.getInputVector();
        return Arrays.asList(
        	    mv.getA1().getAlarmId(),
        	    mv.getA2().getAlarmId(),
                v.getTypeA(),
                v.getTypeB(),
                mv.getA1().getAlarmTime(),
                mv.getA2().getAlarmTime(),
                v.isSameInstance() ? 1 : 0,
                v.isSameParent() ? 1 : 0,
                v.isShareAncestor() ? 1 : 0,
                v.getTimeDifferenceInSeconds(),
                v.getDistanceOnGraph(),
                mv.getStdist(),
                v.getSimilarityOfInventoryObjectIds(),
                v.getSimilarityOfInventoryObjectLabels(),
                ov.areAlarmsRelated() ? 1 : 0
        );
    }

}