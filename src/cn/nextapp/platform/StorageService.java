package cn.nextapp.platform;

import my.mvc.RequestContext;
import my.util.Storage;

/**
 * 文件存储服务
 * @author Winter Lau
 * @date 2010-9-2 上午11:35:56
 */
public class StorageService extends Storage {

	public final static StorageService IMAGES = new StorageService("img");
	
	private String file_path;

	private StorageService(String ext){
		this.file_path = RequestContext.root() + 
				"uploads" + java.io.File.separator + 
				ext + java.io.File.separator;
	}
	
	@Override
	protected String getBasePath() {
		return file_path;
	}
	
}
