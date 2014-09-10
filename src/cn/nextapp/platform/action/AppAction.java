package cn.nextapp.platform.action;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.mail.EmailException;
import org.xmlpull.v1.XmlPullParserException;

import cn.nextapp.platform.SmtpHelper;
import cn.nextapp.platform.StorageService;
import cn.nextapp.platform.beans.App;
import cn.nextapp.platform.beans.URLs;
import cn.nextapp.platform.beans.User;
import cn.nextapp.platform.beans.Version;
import cn.nextapp.platform.http.HttpUtils;
import my.mvc.Annotation;
import my.mvc.RequestContext;
import my.util.ResourceUtils;
import my.view.FormatTool;

/**
 * 网站创建、删除和修改
 * @author Winter Lau
 * @date 2011-12-29 下午10:05:38
 */
public class AppAction {

	public final static String[] IMAGE_FORMATS = {"png"};
	
	public final static int ICO_WIDTH 	= 200;
	public final static int ICO_HEIGHT	= 200;
	
	public final static int LOGO_WIDTH 	= 150;
	public final static int LOGO_HEIGHT	= 40;
	
	public final static int SPLASH_WIDTH 	= 480;
	public final static int SPLASH_HEIGHT	= 800;
	
	/**
	 * 检查版本更新
	 * <?xml version="1.0" encoding="UTF-8"?>
	 * <nextapp>
	 * <version>
	 * 	<id>2</id>
	 *  <name>1.1</name>
	 *  <url>http://www.nextapp.cn/dl/2.apk</url>
	 *  <changelog>增加了上下页功能</changelog>
	 * </version>
	 * </nextapp>
	 * @param ctx
	 * @throws IOException 
	 */
	public void check_version(RequestContext ctx) throws IOException {
		String user_agent = ctx.user_agent();
		try{
			String app_version = StringUtils.split(user_agent, '/')[0];
			String s_ver = app_version.substring("NextApp ".length());
			int version_id = Integer.parseInt(s_ver.substring(s_ver.indexOf('_')+1));
			Version newVer = Version.checkNewVersion(version_id);
			if(newVer != null){
				
			}
			ctx.response().sendError(HttpServletResponse.SC_NOT_MODIFIED);
		}catch(Exception e){
			ctx.response().sendError(HttpServletResponse.SC_BAD_REQUEST);
		}
	}
	
	/**
	 * 新建博客APP
	 * @param ctx
	 * @throws IOException 
	 */
	@Annotation.UserRoleRequired
	@Annotation.JSONOutputEnabled
	public void newblog(RequestContext ctx) throws IOException {	
		App form = ctx.form(App.class);

		//if(App.getAppByName(form.getName()) != null)
		//	throw ctx.error("site_name_exists", form.getName());
		
		//检查表单
		check_home(ctx, form.getHome_url());	
		
		if(StringUtils.isBlank(form.getName()) || form.getName().trim().getBytes().length > 32)
			throw ctx.error("app_name_illegal");
		
		check_plugin(ctx, form.getPlugin_url());
		form.setDomain(new URL(form.getHome_url()).getHost());
		if(App.getAppByDomain(form.getDomain()) != null)//检查网站是否已经注册过
			throw ctx.error("site_name_exists", form.getDomain());
		
		User user = (User)ctx.user();
		form.setType(App.TYPE_BLOG);
		form.setSite(form.getName());
		form.setUser(user.getId());
		int app_id = form.Save();

		Version.init(app_id);
		ctx.output_json("guid", form.getGuid());
	}

	/**
	 * 编辑app
	 * @param ctx
	 * @throws IOException
	 */
	@Annotation.UserRoleRequired
	@Annotation.JSONOutputEnabled
	public void editapp(RequestContext ctx) throws Exception {	
		App form = ctx.form(App.class);
		User user = (User)ctx.user();
		App app = App.getAppByGuid(form.getGuid());
		if(app == null || app.getUser() != user.getId())
			throw ctx.error("app_not_found");
		
		if(StringUtils.isBlank(form.getName()) || form.getName().trim().getBytes().length > 16)
			throw ctx.error("app_name_illegal");
		
		//检查表单
		if(!StringUtils.equalsIgnoreCase(form.getHome_url(), app.getHome_url()))
			check_home(ctx, form.getHome_url());
		if(!StringUtils.equalsIgnoreCase(form.getPlugin_url(), app.getPlugin_url()))
			check_plugin(ctx, form.getPlugin_url());

		File icoFile = ctx.file("icoFile");
		if(icoFile != null){
			_CheckImg(ctx, ResourceUtils.ui("img_ico"), icoFile, ICO_WIDTH, ICO_HEIGHT);
			app.setIco(StorageService.IMAGES.save(icoFile));
		}
		File logoFile = ctx.file("logoFile");
		if(logoFile != null){
			_CheckImg(ctx, ResourceUtils.ui("img_logo"), logoFile, LOGO_WIDTH, LOGO_HEIGHT);
			app.setLogo(StorageService.IMAGES.save(logoFile));
		}
		File welcomeFile = ctx.file("welcomeFile");
		if(welcomeFile != null){
			_CheckImg(ctx, ResourceUtils.ui("img_splash"), welcomeFile, SPLASH_WIDTH, SPLASH_HEIGHT);
			app.setWelcome(StorageService.IMAGES.save(welcomeFile));
		}
		
		app.setName(form.getName());
		app.setOutline(form.getOutline());
		app.setHome_url(form.getHome_url());
		app.setPlugin_url(form.getPlugin_url());
		
		app.Update();
		_NotifyToAdministrator(app);
		
		ctx.output_json("guid", app.getGuid());
	}
	
	/**
	 * 上传App图片
	 * @param ctx
	 * @throws IOException 
	 */
	@Annotation.UserRoleRequired
	@Annotation.JSONOutputEnabled
	public void upload_pic(RequestContext ctx) throws Exception {
		
		File icoFile = ctx.file("ico");
		_CheckImg(ctx, ResourceUtils.ui("img_ico"), icoFile, ICO_WIDTH, ICO_HEIGHT);
		File logoFile = ctx.file("logo");
		_CheckImg(ctx, ResourceUtils.ui("img_logo"), logoFile, LOGO_WIDTH, LOGO_HEIGHT);
		File welcomeFile = ctx.file("welcome");
		_CheckImg(ctx, ResourceUtils.ui("img_splash"), welcomeFile, SPLASH_WIDTH, SPLASH_HEIGHT);
		String guid = ctx.param("guid");
		
		User user = (User)ctx.user();
		
		App app = App.getAppByGuid(guid);
		if(app == null || app.getUser() != user.getId())
			throw ctx.error("app_not_found");
		
		app.setIco(StorageService.IMAGES.save(icoFile));
		app.setLogo(StorageService.IMAGES.save(logoFile));
		app.setWelcome(StorageService.IMAGES.save(welcomeFile));

		app.UploadPics();
		_NotifyToAdministrator(app);
		
		ctx.output_json("guid", app.getGuid());
	}
	
	private void _NotifyToAdministrator(App app) throws EmailException {
		String title = ResourceUtils.getString("ui", "app_notify_title", app.getDomain());
		String html = "<a href='" + app.getHome_url() + "'>" + app.getName() + "</a>";
		SmtpHelper.sendToAdministrator(title, html);
	}
	
	/**
	 * 检查图片是否符合要求
	 * @param ctx
	 * @param what
	 * @param img
	 * @param width
	 * @param height
	 * @param format
	 * @throws IOException 
	 */
	private void _CheckImg(RequestContext ctx, String what, File img, int width, int height) throws IOException {
		if(img == null)
			throw ctx.error("img_file_empty",what);
		String ext = StringUtils.lowerCase(FilenameUtils.getExtension(img.getName()));
		if(!ArrayUtils.contains(IMAGE_FORMATS, ext))
			throw ctx.error("img_fmt_illegal",what);
		BufferedImage bi = (BufferedImage)ImageIO.read(img);
		if(bi.getWidth() != width || bi.getHeight() != height)
			throw ctx.error("img_size_illegal", what, bi.getWidth(), bi.getHeight(), width, height);		
	}

	/**
	 * 检查首页是否能访问
	 * @param ctx
	 * @param url
	 */
	private void check_home(RequestContext ctx, String url) {
		if(!FormatTool.is_link(url))
			throw ctx.error("url_illegal");
		if(!HttpUtils.isUrlAccessable(url))
			throw ctx.error("url_unreachable");
	}
	
	/**
	 * 检查插件地址是否有效
	 * @param ctx
	 * @param url
	 */
	private void check_plugin(RequestContext ctx, String url) {
		if(!FormatTool.is_link(url))
			throw ctx.error("url_illegal");
		//解析每一个接口
		try {
			URLs.parse(HttpUtils.httpGet(url));
		} catch (XmlPullParserException e) {
			throw ctx.error("plugin_url_xml_parse_error", e.toString());
		} catch (Exception e){
			throw ctx.error("plugin_url_get_error", e.toString());
		}
	}
}
