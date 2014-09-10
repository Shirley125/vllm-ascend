## auto build script
export JAVA_HOME=/usr/local/jdk
export NEXTAPP_HOME=/nextapp/platform
cd $NEXTAPP_HOME
$JAVA_HOME/bin/java -cp "$NEXTAPP_HOME/packages/ant-launcher.jar:$NEXTAPP_HOME/packages/ant.jar:$JAVA_HOME/lib/tools.jar" org.apache.tools.ant.launch.Launcher -lib $NEXTAPP_HOME/packages $*
