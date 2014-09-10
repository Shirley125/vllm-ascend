## auto build script
export JAVA_HOME=/usr/local/jdk
export NEXTAPP_HOME=/nextapp/platform
cd $NEXTAPP_HOME
echo "`date` : begin to check building..."
$JAVA_HOME/bin/java -Djava.ext.dirs=$NEXTAPP_HOME/packages:$NEXTAPP_HOME/webapp/WEB-INF/lib -cp $NEXTAPP_HOME/webapp/WEB-INF/classes cn.nextapp.platform.BuildDaemon $*
