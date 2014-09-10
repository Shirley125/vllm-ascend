package cn.nextapp.platform.action;

import java.util.Arrays;

import javax.servlet.http.HttpServletResponse;

import my.mvc.Annotation;
import my.mvc.RequestContext;

import org.apache.commons.lang.StringUtils;
import org.brickred.socialauth.AuthProvider;
import org.brickred.socialauth.Profile;
import org.brickred.socialauth.SocialAuthConfig;
import org.brickred.socialauth.SocialAuthManager;
import org.brickred.socialauth.exception.UserDeniedPermissionException;
import org.brickred.socialauth.util.OAuthConfig;
import org.brickred.socialauth.util.SocialAuthUtil;

import cn.nextapp.platform.UserLoginManager;
import cn.nextapp.platform.beans.Openid;
import cn.nextapp.platform.beans.User;

/**
 * 使用SocialAuth支持的登录方法
 * @author Winter Lau
 * @date 2011-12-28 下午2:38:28
 */
public class OpenidAction {

	private final static String SESSION_KEY = "SocialAuth_Manager";
	
	private final static SocialAuthConfig config;
	
	static {
		// Create an instance of SocialAuthConfgi object
		config = SocialAuthConfig.getDefault();
		
		// load configuration. By default load the configuration from oauth_consumer.properties.
		// You can also pass input stream, properties object or properties file name.
		try {
			config.load();
			//加载新浪微博和QQ登录
			for(String provider_name : Arrays.asList("weibo", "qq")){
				String className = config.getApplicationProperties().getProperty(provider_name);
				String consumer_key = config.getApplicationProperties().getProperty(provider_name+".consumer_key");
				String consumer_secret = config.getApplicationProperties().getProperty(provider_name+".consumer_secret");
				OAuthConfig c = new OAuthConfig(consumer_key, consumer_secret);
				c.setProviderImplClass(Class.forName(className));
				config.addProviderConfig(provider_name, c);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * 使用OpenID登录
	 * @param ctx
	 * @throws Exception 
	 */
	public void request(RequestContext ctx) throws Exception {
		String provider_name = ctx.param("provider");
		// Create an instance of SocialAuthManager and set config
		SocialAuthManager manager = new SocialAuthManager();

		manager.setSocialAuthConfig(config);
		String returnToUrl = ctx.request().getScheme() + "://"+ ctx.request().getServerName();
		returnToUrl += "/action/openid/login";
		String url = manager.getAuthenticationUrl(provider_name, returnToUrl);
		ctx.session(true).setAttribute(SESSION_KEY, manager);
		ctx.redirect(url);
	}

	/**
	 * 请求OpenID登录绑定
	 * @param ctx
	 * @throws Exception 
	 */
	@Annotation.UserRoleRequired
	public void request_bind(RequestContext ctx) throws Exception {
		String provider_name = ctx.param("provider");
		// Create an instance of SocialAuthManager and set config
		SocialAuthManager manager = new SocialAuthManager();
		manager.setSocialAuthConfig(config);
		String bindUrl = ctx.request().getScheme() + "://"+ ctx.request().getServerName();
		bindUrl += "/action/openid/add_bind" ;
		String url = manager.getAuthenticationUrl(provider_name, bindUrl);
		ctx.session(true).setAttribute(SESSION_KEY, manager);
		ctx.redirect(url);
	}
	
	/**
	 * 添加OpenID的绑定
	 * @param ctx
	 * @throws Exception
	 */
	@Annotation.UserRoleRequired
	public void add_bind(RequestContext ctx) throws Exception {	
		User user = (User)ctx.user();
		Profile p = getProfile(ctx);
		if(p == null){
			ctx.error(HttpServletResponse.SC_BAD_REQUEST, "Profile is Null");
			return ;
		}
		
		String provider_name = p.getProviderId().toLowerCase();
		
		Openid openid = Openid.getOpenid(p.getEmail(), provider_name);
		if(openid != null && openid.getUser() != user.getId()) 
			throw ctx.error("openid_used_by_another_user");		

		Openid.saveOpenId(user.getId(), p.getEmail(), provider_name , getName(p));
		
		ctx.redirect("/my/bind_openid");
	}

	/**
	 * 验证OpenID登录的有效性
	 * @param ctx
	 * @throws Exception
	 */
	public void login(RequestContext ctx) throws Exception {
		// get profile
		Profile p = getProfile(ctx);

		if(p == null){
			ctx.error(HttpServletResponse.SC_BAD_REQUEST, "Profile is Null");
			return ;
		}
		
		String provider_name = p.getProviderId().toLowerCase();
		
		// you can obtain profile information
		Openid openid = Openid.loginViaOpenId(p.getEmail(), provider_name);
		if(openid != null){
			UserLoginManager.save(ctx, openid.getUser());
			ctx.redirect("/my");
		}
		else {
			//首次使用的用户
			User user = User.newUser(getName(p), p.getEmail());
			Openid.saveOpenId(user.getId(), user.getEmail(), provider_name , user.getName());
			UserLoginManager.save(ctx, user.getId());
			ctx.redirect("/newuser");
		}
	}
		
	private Profile getProfile(RequestContext ctx) throws Exception {
		try{
			// get the social auth manager from session
			SocialAuthManager manager = (SocialAuthManager)ctx.sessionAttr(SESSION_KEY);
			// call connect method of manager which returns the provider object.
			// Pass request parameter map while calling connect method.
			if(manager == null)
				return null;
			AuthProvider provider = manager.connect(SocialAuthUtil.getRequestParametersMap(ctx.request()));
	
			// get profile
			return provider.getUserProfile();
		}catch(UserDeniedPermissionException e){
			throw ctx.error("UserDeniedPermissionException");
		}
		
	}
	
	/**
	 * 返回用户姓名
	 * @param p
	 * @return
	 */
	private String getName(Profile p) {
		String fn = p.getFirstName();
		String ln = p.getLastName();
		String email = p.getEmail();
		String name = "";
		if(StringUtils.isNotBlank(fn) && StringUtils.isNotBlank(ln)){
			if(StringUtils.equals(fn, ln))
				name = fn;
			else{
				if(fn.length() < ln.length())
					name = fn + ln;
				else
					name = ln + fn;
			}
		}
		else{
			if(StringUtils.isNotBlank(ln))
				name = ln;
			if(StringUtils.isNotBlank(fn))
				name += fn;
		}
		if(StringUtils.isBlank(name))
			name = email.substring(0, email.indexOf('@'));
		
		return name;
	}
	
}
