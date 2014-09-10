package cn.nextapp.platform.action;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;

import cn.nextapp.platform.NextApp;
import cn.nextapp.platform.beans.App;
import cn.nextapp.platform.beans.Version;
import my.mvc.RequestContext;
import my.util.StringUtils;

/**
 * 软件下载
 * @author Winter Lau
 * @date 2012-1-14 下午2:43:58
 */
public class DlAction {

	/**
	 * 下载app
	 * @param ctx
	 * @throws IOException 
	 */
	public void index(RequestContext ctx) throws IOException {
		String guid = ctx.param("app");
		int repo = ctx.param("ver", 0);
		
		App app = App.getAppByGuid(guid);
		if(app == null || app.getStatus()<App.STATUS_AUDIT_PASSED){
			ctx.not_found();
			return;
		}
		List<Version> versions = app.versions();
		for(Version ver : versions) {
			if(ver.getVersion() == repo) {
				_DownloadFile(ctx, ver);
				ver.incDlCount(1);
				return ;
			}
		}
		ctx.not_found();
	}
	
	/**
	 * 下载文件
	 * @param ctx
	 * @param ver
	 * @throws IOException
	 */
	private void _DownloadFile(RequestContext ctx, Version ver) throws IOException {
		File app = new File(NextApp.getBuildTarget() + File.separator + ver.getApp_path());
		ctx.response().setContentLength((int)app.length());
		ctx.response().setContentType("application/octet-stream");
		App theApp = App.INSTANCE.Get(ver.getApp());
		String fn = getIdentOfDomain(theApp.getDomain()) + "." + FilenameUtils.getExtension(app.getName());
		ctx.header("Content-Disposition","attachment; filename=" + fn);
		FileInputStream fis = new FileInputStream(app);
		try{
			IOUtils.copy(fis, ctx.response().getOutputStream());
		}finally{
			IOUtils.closeQuietly(fis);
		}
	}
	
	private static String getIdentOfDomain(String domain) {
		int dotC = StringUtils.countMatches(domain, DOT);
		if(dotC == 0)
			return domain;
		if(dotC == 1)
			return StringUtils.substringBefore(domain, DOT);
		if(dotC == 2)
			return StringUtils.substringBetween(domain, DOT);
		int len = domain.length();
		for(String key : DOMAINS)
			domain = StringUtils.remove(domain, key);
		if(domain.length() == len)
			return domain;
		return getIdentOfDomain(domain);
	}
	
	public static void main(String[] args) {
		String domain = "www.javayou.com.cn";
		System.out.println(getIdentOfDomain(domain));
	}
	
	private final static String[] DOMAINS = {".com",".net",".org",".cn","www.","blog."};
	private final static String DOT = ".";
}
