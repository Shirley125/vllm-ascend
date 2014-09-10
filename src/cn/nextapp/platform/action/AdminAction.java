package cn.nextapp.platform.action;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

import my.mvc.Annotation;
import my.mvc.RequestContext;
import my.util.ResourceUtils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;

import cn.nextapp.platform.NextApp;
import cn.nextapp.platform.StorageService;
import cn.nextapp.platform.beans.App;
import cn.nextapp.platform.beans.Repository;
import cn.nextapp.platform.beans.User;
import cn.nextapp.platform.beans.Version;

/**
 * 后台管理
 * @author Winter Lau
 * @date 2012-1-14 上午11:57:53
 */
public class AdminAction {

	private void checkPermission(RequestContext ctx) {
		User user = (User)ctx.user();
		if(!user.IsAdmin())
			throw ctx.error("user_permission_not_enough");
	}
	
	/**
	 * 删除App
	 */
	@Annotation.UserRoleRequired
	public void delete_app(RequestContext ctx) throws IOException {
		checkPermission(ctx);
		String guid = ctx.param("guid");
		App app = App.getAppByGuid(guid);
		if(app != null)
			app.Delete();
	}

	/**
	 * app审批
	 * @param ctx
	 * @throws IOException
	 */
	@Annotation.UserRoleRequired
	public void audit_app(RequestContext ctx) throws IOException {
		checkPermission(ctx);
		boolean pass = Boolean.valueOf(ctx.param("pass","true"));
		String guid = ctx.param("guid");
		App app = App.getAppByGuid(guid);
		if(app == null)
			throw ctx.error("app_not_found");
		if(pass)
			approve(ctx, app);
		else
			deny(ctx, app);
	}
	
	/**
	 * app审批通过
	 * @param ctx
	 * @throws IOException 
	 */
	private void approve(RequestContext ctx, App app) throws IOException {
		app.approve();
		for(Version ver : app.versions()){
			_GenerateBuildFiles(app, ver);
		}
	}

	/**
	 * 审批不通过
	 * @param ctx
	 */
	private void deny(RequestContext ctx, App app) {
		String reason = ctx.param("reason");
		if(StringUtils.isBlank(reason))
			throw ctx.error("app_deny_reason_empty");
		app.deny(reason);
	}
	
	/**
	 * 生成待编译的文件
	 * @param app
	 * @param ver
	 * @throws IOException 
	 */
	private void _GenerateBuildFiles(App app, Version ver) throws IOException {
		Repository repo = new Repository().Get(ver.getVersion());
		//1. 复制程序
		File destDir = new File(NextApp.getBuildSource().getPath() + File.separator + "app_"+app.getId()+"_"+ver.getId());
		FileUtils.copyDirectory(new File(repo.getSrc()), destDir);
		//2. 生成 build.properties
		Properties props = new Properties();
		props.setProperty("app_name", app.getName());		
		props.setProperty("app_version_name", repo.getVersion());
		props.setProperty("app_version_code", String.valueOf(ver.getId()));
		props.setProperty("app_product_id", app.getGuid());
		if(ver.getClient_type() == NextApp.OS_ANDROID)
			props.setProperty("app_about_intro", app.getOutline().replace("\n", "\\n"));
		else
			props.setProperty("app_about_intro", app.getOutline());
		props.setProperty("app_about_website", app.getHome_url());
		props.setProperty("app_api_url", app.getPlugin_url());
		//TODO: 根据 ver.client_type 来生成不同的图片大小
		props.setProperty("app_imgpath_icon",StorageService.IMAGES.readFile(app.getIco()).getPath());
		props.setProperty("app_imgpath_logo",StorageService.IMAGES.readFile(app.getLogo()).getPath());
		props.setProperty("app_imgpath_welcome",StorageService.IMAGES.readFile(app.getWelcome()).getPath());

		String package_name = "cn.nextapp.app.blog.app_" + app.getId();
		props.setProperty("app_package_name", package_name);
		props.setProperty("app_package_path", StringUtils.replace(package_name, ".", "/"));
		
		File propsFile = new File(destDir + File.separator + "build.properties");
		FileOutputStream fos = new FileOutputStream(propsFile);
		try{
			props.store(fos, "Build.properties for " + ResourceUtils.ui("client_type_"+ver.getClient_type()));
		}finally{
			fos.close();
		}
	}	
	
}
