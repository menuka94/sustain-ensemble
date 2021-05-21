package org.sustain.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.sustain.handlers.EnsembleQueryHandler;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CountyClusters {

    private static final Logger log = LogManager.getLogger(CountyClusters.class);

    public static Map<String, List> extractCountyGroups(String filePath) {
        HashMap<String, List> countyMap = new HashMap<String, List>();
        Map<Integer, String> clusterIDToCentroidMap = new HashMap<Integer, String>();
        HashMap<Integer, List> clusterIDToDependents = new HashMap<Integer, List>();
        try {

            String c = new String(Files.readAllBytes(Paths.get(filePath)));

            String[] lines = c.split("\n");
            for(String line: lines) {
                if(line.contains("cluster"))
                    continue;

                String tokens[] = line.split(",");
                int clusterID = Integer.valueOf(tokens[0]);
                String gisJoin = tokens[1];
                boolean isCenter = Boolean.valueOf(tokens[2]);

                if(isCenter) {
                    if(clusterIDToCentroidMap.get(clusterID) != null) {
                        System.out.println("CLUSTERID:"+clusterID+" HAS MULTIPLE CENTROIDS");
                    }
                    clusterIDToCentroidMap.put(clusterID, gisJoin);
                } else {
                    List<String> dependentCounties = new ArrayList<String>();
                    if(clusterIDToDependents.get(clusterID) != null) {
                        dependentCounties = clusterIDToDependents.get(clusterID);
                    }
                    dependentCounties.add(gisJoin);
                    clusterIDToDependents.put(clusterID, dependentCounties);
                }

            }

            for(int clusterID: clusterIDToCentroidMap.keySet()) {
                String centroidCounty = clusterIDToCentroidMap.get(clusterID);
                List<String> dependentCounties = clusterIDToDependents.get(clusterID);
                countyMap.put(centroidCounty, dependentCounties);
            }

        } catch(Exception e) {
            log.error("ERROR PARSING CLUSTER FILE: "+e);
        }
        return countyMap;
    }

    public static List<List<String>> makeSoloBatch(String parentGisJoin, int batchSize) {
        List<List<String>> batches = new ArrayList<>();

        batches.add(new ArrayList<>());

        batches.get(batches.size() - 1).add(parentGisJoin);


        StringBuilder batchLog = new StringBuilder(
                String.format(">>>PARENT... %d batches for %d GISJoins\n", batches.size(), 1)
        );
        for (int i = 0; i < batches.size(); i++) {
            batchLog.append(String.format("\tBatch %d size: %d\n", i, batches.get(i).size()));
        }
        log.info(batchLog.toString());
        return batches;
    }
}
