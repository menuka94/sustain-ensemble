package org.sustain.util;

import org.apache.logging.log4j.Logger;

public class FancyLogger {

    public static void fancy_logging(String msg, Logger log){

        String logStr = "\n============================================================================================================\n";
        logStr+=msg;
        logStr+="\n============================================================================================================";

        log.info(logStr);
    }
}
