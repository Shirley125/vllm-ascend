package cn.nextapp.platform;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;
import org.apache.tools.ant.DefaultLogger;
import org.apache.tools.ant.Project;
import org.apache.tools.ant.ProjectHelper;

import cn.nextapp.platform.beans.Version;

/**
 * 监控目录并进行程序构建
 * @author Winter Lau
 * @date 2012-1-4 下午1:01:59
 */
public class BuildDaemon {

	private static DefaultLogger g_logger;
	private static MessageFormat FMT_APP_NAME = new MessageFormat("app_{0,number,integer}_{1,number,integer}");
	
	/**
	 * 程序入口
	 * @param args
	 * @throws ParseException 
	 */
	public static void main(String[] args) throws IOException, ParseException {
				
		File srcPath = NextApp.getBuildSource();
		File destPath = NextApp.getBuildTarget();
				
		File[] apps = srcPath.listFiles(new FileFilter(){
			@Override
			public boolean accept(File f) {
				if(!f.isDirectory())
					return false;
				try {
					FMT_APP_NAME.parse(f.getName());
				} catch (java.text.ParseException e) {
					return false;
				}
				String xml = f.getPath() + File.separator + "build.xml";				
				return new File(xml).exists();
			}}
		);
		
		g_logger = buildLogger(Project.MSG_INFO);
		
		if(!destPath.exists() || !destPath.isDirectory())
			throw new FileNotFoundException(destPath.getPath());
		
		//目录迁移
		List<File> newApps = new ArrayList<File>();
		for(File app : apps){
			File destDir = new File(destPath.getPath()+File.separator+app.getName());
			if(destDir.exists())
				FileUtils.forceDelete(destDir);
			FileUtils.moveDirectory(app, destDir);
			newApps.add(destDir);
		}
		
		//移动完所有项目后进行构建
		for(File app : newApps){
			Object[] t_args;
			try {
				t_args = FMT_APP_NAME.parse(app.getName());
				int version_id = ((Number)t_args[1]).intValue();
				Version ver = Version.INSTANCE.Get(version_id);
				buildApp(ver, app + File.separator + "build.xml");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * 构建应用程序
	 * @param build_xml_path
	 * @throws Exception 
	 */
	private static void buildApp(Version ver, String build_xml_path) throws Exception {
		try {
			ver.beginBuild();
			File buildFile = new File(build_xml_path);
			
			Project p = new Project();
			p.addBuildListener(g_logger);
			p.init();
			ProjectHelper helper = ProjectHelper.getProjectHelper();
			helper.parse(p, buildFile);
			p.executeTarget(p.getDefaultTarget());	
			//拷贝构建后的程序文件到指定目录
			String app_path = _getAppPath(ver, buildFile.getParentFile());
			//构建成功，更新数据库并发送邮件给用户		
			ver.endBuild(true, app_path);
		} catch (Exception e) {
			ver.endBuild(false, null);
			throw e;
		} 
	}
	
	/**
	 * 拷贝生成的app到指定目录
	 * @return 返回 app 的路径信息
	 */
	private static String _getAppPath(Version ver, File path) {
		switch(ver.getClient_type()){
		case NextApp.OS_ANDROID:
			return path.getName() + File.separator + "NextApp.apk";
		case NextApp.OS_WP7:
			return path.getName() + File.separator + "NextApp.xap";
		case NextApp.OS_IPHONE:
			return path.getName() + File.separator + "NextApp.ipa";
		default:
		}
		return null;
	}
	
	/**
	 * 日志记录
	 * @param log_path
	 * @param log_level
	 * @return
	 * @throws IOException
	 */
	private static DefaultLogger buildLogger(int log_level) throws IOException {
		DefaultLogger consoleLogger = new DefaultLogger();
		consoleLogger.setErrorPrintStream(System.err);
		consoleLogger.setOutputPrintStream(System.out);
		consoleLogger.setMessageOutputLevel(log_level);
		return consoleLogger;
	}
	
}
