@echo off
java -Djava.ext.dirs=packages;webapp/WEB-INF/lib -cp "webapp/WEB-INF/classes" cn.nextapp.platform.BuildDaemon %1 %2 %3 %4 %5 %6 %7 %8