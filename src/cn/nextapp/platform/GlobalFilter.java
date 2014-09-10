package cn.nextapp.platform;

import javax.servlet.FilterConfig;
import javax.servlet.ServletException;

import cn.nextapp.platform.beans.User;

import my.mvc.RequestContext;
import my.mvc.URLMappingFilter;

/**
 * 全局过滤器
 * @author Winter Lau
 * @date 2011-12-28 下午1:57:23
 */
public class GlobalFilter extends URLMappingFilter {

	@Override
	public void init(FilterConfig cfg) throws ServletException {
		super.init(cfg);
	}

	@Override
	protected void beforeFilter(RequestContext ctx) {
		int userid = UserLoginManager.get(ctx);
		if(userid > 0){
			User user = new User().Get(userid);
			if(user != null)
				ctx.request().setAttribute(RequestContext.GLOBAL_USER_KEY, user);
		}
	}

	@Override
	public void destroy() {
		super.destroy();
	}

}
