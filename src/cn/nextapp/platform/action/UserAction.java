package cn.nextapp.platform.action;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang.StringUtils;

import cn.nextapp.platform.SmtpHelper;
import cn.nextapp.platform.UserLoginManager;
import cn.nextapp.platform.beans.Openid;
import cn.nextapp.platform.beans.User;
import cn.nextapp.platform.toolbox.LinkTool;
import my.mvc.*;
import my.util.ResourceUtils;
import my.view.FormatTool;
import my.view.VelocityHelper;

/**
 * 用户相关Action
 * @author Winter Lau
 * @date 2011-12-28 下午5:04:01
 */
public class UserAction {

	/**
	 * 联系我们
	 * @param ctx
	 */
	@Annotation.PostMethod
	public void contact_us(RequestContext ctx) {
		Contact form = ctx.form(Contact.class);
		form.check(ctx);
		form.send(ctx, form);
	}
	
	public static class Contact {
		
		/**
		 * 检查表单
		 * @param ctx
		 */
		public void check(RequestContext ctx) {
			if(StringUtils.isBlank(this.name))
				throw ctx.error("contact_name_empty");
			if(StringUtils.isBlank(this.tel))
				throw ctx.error("contact_tel_empty");
			if(!FormatTool.is_email(this.email))
				throw ctx.error("contact_email_illegal");
			if(StringUtils.isBlank(this.message))
				throw ctx.error("contact_message_empty");
		}
		
		/**
		 * 通过邮件发送留言
		 * @param ctx
		 */
		public void send(RequestContext ctx, final Contact form) {
			String title = ResourceUtils.getString("error", "contact_email_title", form.name);
			String tpl = "/WEB-INF/pages/mail/contact_msg.html";
			String html;
			try {
				html = VelocityHelper.execute(tpl, new HashMap<String, Object>(){{
					put("form",form);
				}});
				SmtpHelper.send(SmtpHelper.getAdministrator(), title, html);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		private String name;
		private String tel;
		private String email;
		private String message;
		public void setName(String name) { this.name = name; } 
		public void setTel(String tel) { this.tel = tel; }
		public void setEmail(String email) { this.email = email; }
		public void setMessage(String message) { this.message = message; }
	}
	
	/**
	 * 修改用户名和邮箱
	 * @param ctx
	 */
	@Annotation.UserRoleRequired
	@Annotation.PostMethod
	public void updateInfo(RequestContext ctx) {
		User loginUser = (User)ctx.user();
		User form = ctx.form(User.class);
		if(!FormatTool.is_email(form.getEmail()))
			throw ctx.error("email_illegal");
		loginUser.setName(form.getName());
		loginUser.setEmail(form.getEmail());
		if(StringUtils.isNotBlank(form.getPhone()))
			loginUser.setPhone(form.getPhone());
		//TODO: 验证输入值
		loginUser.Update();
	}
	
	/**
	 * 退出登录
	 * @param ctx
	 * @throws IOException 
	 */
	public void logout(RequestContext ctx) throws IOException {
		UserLoginManager.delete(ctx);
		ctx.redirect(LinkTool.root());
	}
	
	/**
	 * 删除OpenID 帐号
	 * @param ctx
	 */
	@Annotation.UserRoleRequired
	public void deleteOpenid(RequestContext ctx) {
		int id = ctx.id();
		User user = (User)ctx.user();
		List<Openid> openids = user.openids();
		if(openids.size() == 1)
			throw ctx.error("openid_delete_not_allow");
		for(Openid bean : openids){
			if(bean.getId() == id){
				bean.Delete();
				return ;
			}
		}
		throw ctx.error("openid_not_found");
	}
}
