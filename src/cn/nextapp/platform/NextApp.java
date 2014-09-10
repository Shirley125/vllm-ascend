package cn.nextapp.platform;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

import cn.nextapp.platform.action.AppAction;
import cn.nextapp.platform.beans.App;
import cn.nextapp.platform.beans.Version;

import my.mvc.RequestContext;

/**
 * 定义一些系统产量
 * @author Winter Lau
 * @date 2011-12-30 下午3:49:48
 */
public class NextApp {

	public final static byte OS_ANDROID 		= 0x01;
	public final static byte OS_IPHONE			= 0x02;		
	public final static byte OS_WP7 			= 0x03;
	public final static byte OS_IPAD			= 0x04;
	public final static byte OS_ANDROID_PAD		= 0x05;
	
	private final static String rootPath;

	private final static File build_source_path; 
	private final static File build_target_path; 
	
	static {
		InputStream fis = AppAction.class.getResourceAsStream("/nextapp.config");
		Properties props = new Properties();
		try {
			props.load(fis);
			String sBUILD_DIR = props.getProperty("BUILD_SOURCE");
			build_source_path = _AutoInitPath(sBUILD_DIR);
			build_target_path = _AutoInitPath(props.getProperty("BUILD_TARGET"));
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		rootPath = _GetRootPath();
	}
	
	public static File getBuildSource(){
		return build_source_path;
	}

	public static File getBuildTarget(){
		return build_target_path;
	}
	
	public static String getRootPath() {
		return rootPath;
	}

	private final static String _GetRootPath() {
		try{
			String root = RequestContext.class.getResource("/nextapp.config").getFile();
			root = new File(root).getParentFile().getParentFile().getParentFile().getCanonicalPath();
			root += File.separator;
			return root;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static File _AutoInitPath(String path) throws FileNotFoundException {
		File fFile = new File(path);
		if(!fFile.exists() || !fFile.isDirectory())
			throw new FileNotFoundException(path);
		return fFile;
	}
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		if("regenerate_app".equalsIgnoreCase(args[0])){
			List<App> apps = (List<App>)App.INSTANCE.List(1, Integer.MAX_VALUE);
			for(App app : apps){				
				Version.init(app.getId());
			}
		}
	}
	
}
