package cn.nextapp.platform.toolbox;

import cn.nextapp.platform.beans.App;

/**
 * Toolbox
 * @author Winter Lau
 * @date 2012-1-4 下午6:08:48
 */
public class NextAppTools {

	public static App app(int app_id) {
		return App.INSTANCE.Get(app_id);
	}
	
	public static App app(String guid) {
		return App.getAppByGuid(guid);
	}
	
}
