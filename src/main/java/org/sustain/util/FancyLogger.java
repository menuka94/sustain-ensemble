package org.sustain.util;

import org.apache.logging.log4j.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FancyLogger {

    public static String base_path = "/s/parsons/b/others/sustain/sustain-transfer-learning/";

    public static String fancy_logging(String msg, Logger log){

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================\n";

        if(log != null)
            log.info(logStr);
        return logStr;
    }

    public static String fancy_logging(String msg){

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================\n";

        return logStr;
    }

    public static String write_out(String msg, String filename){

        File file = new File(base_path+filename);

        /* This logic will make sure that the file
         * gets created if it is not present at the
         * specified location*/
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                System.out.println("**********ERROR PRINTING*********"+filename);
                e.printStackTrace();
            }
        }

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================\n";

        String fileabspath = base_path+filename;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileabspath,true))) {
            writer.append(msg);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return logStr;
    }
}
